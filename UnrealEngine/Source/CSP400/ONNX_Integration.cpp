#include "ONNX_Integration.h"

AONNX_Integration::AONNX_Integration()
{
    PrimaryActorTick.bCanEverTick = true;
}

void AONNX_Integration::BeginPlay()
{
    Super::BeginPlay();

    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("SPIDER: ModelData asset is missing."));
        return;
    }

    // Initialize buffer sizes to prevent memory crashes during inference
    PreviousActions.Init(0.0f, 24);
    PreviousJointAngles.Init(0.0f, 24);
    ObservationsBuffer.Init(0.0f, 132);
    BevDataBuffer.Init(0.0f, 12288);  // [1, 3, 64, 64]
    HeightmapBuffer.Init(0.0f, 4096); // [1, 1, 64, 64]
    NavDataBuffer.Init(0.0f, 1089);
    SmoothedActions.Init(0.0f, 24);

    SpiderMesh = FindComponentByClass<USkeletalMeshComponent>();
    if (!SpiderMesh) return;

    SpiderMesh->SetSimulatePhysics(true);

    BoneNames.Empty();
    ControlNames.Empty();

    // Map out the hexapod skeleton
    for (int32 LegIdx = 0; LegIdx < 6; LegIdx++)
    {
        BoneNames.Add(FName(*FString::Printf(TEXT("hip_joint_%d"), LegIdx)));
        BoneNames.Add(FName(*FString::Printf(TEXT("upper_joint_%d"), LegIdx)));
        BoneNames.Add(FName(*FString::Printf(TEXT("middle_joint_%d"), LegIdx)));
        BoneNames.Add(FName(*FString::Printf(TEXT("lower_joint_%d"), LegIdx)));
    }

    PhysicsControl = FindComponentByClass<UPhysicsControlComponent>();
    if (!PhysicsControl) return;

    SpiderMesh->RecreatePhysicsState();

    // Generate hardware muscles for all 24 joints
    for (int32 i = 0; i < BoneNames.Num(); i++)
    {
        FName BoneName = BoneNames[i];
        FName ParentBone = SpiderMesh->GetParentBone(BoneName);
        int JointType = i % 4; // 0=Hip, 1=Upper, 2=Middle, 3=Lower

        if (ParentBone != NAME_None)
        {
            FPhysicsControlData ControlData;
            ControlData.LinearStrength = 0.0f;
            ControlData.MaxTorque = 0.0f;

            if (JointType == 0)
            {
                // Hips mainly handle lateral sweeps
                ControlData.AngularStrength = 1500.0f;  
                ControlData.AngularDampingRatio = 5.0f; 
            }
            else 
            {
                // Knees need massive torque to bench-press the 20kg chassis
                ControlData.AngularStrength = 3000.0f;   
                ControlData.AngularDampingRatio = 5.0f; 
            }

            FPhysicsControlTarget ControlTarget;

            FName MuscleName = PhysicsControl->CreateControl(
                SpiderMesh, ParentBone,
                SpiderMesh, BoneName,
                ControlData, ControlTarget,
                NAME_None, BoneName.ToString());

            ControlNames.Add(MuscleName);
        }
    }

    ActionOutputBuffer.SetNum(24);
    GruPersistent.SetNumZeroed(256);
    GruUpdated.SetNumZeroed(256);

    // Load the ONNX model into Unreal's NNE CPU runtime
    TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));

    if (Runtime.IsValid())
    {
        TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);
        if (Model.IsValid())
        {
            ModelInstance = Model->CreateModelInstanceCPU();
        }
    }
}

void AONNX_Integration::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    TimeSinceLastInference += DeltaTime;

    // Only tick the brain at 50Hz to match the training environment
    if (TimeSinceLastInference >= InferenceInterval)
    {
        UpdateHeightMap();
        UpdateBEVData(MyRenderTarget);
        UpdateObservations();

        RunSpiderInference();
        SpiderMesh->WakeAllRigidBodies();
        
        ApplyActionsToConstraints();

        TimeSinceLastInference = 0.0f;

        // Apply external directional force to assist with locomotion (simulating intent)
        FVector RobotLocation = SpiderMesh->GetComponentLocation();
        APawn* PlayerPawn = UGameplayStatics::GetPlayerPawn(GetWorld(), 0);

        if (!PlayerPawn) return;

        FVector TargetLoc = (TargetMode == ESpiderTargetMode::Player) ? PlayerPawn->GetActorLocation() : CurrentWaypoint;

        FVector DirectionToTarget = (TargetLoc - RobotLocation).GetSafeNormal();
        DirectionToTarget.Z = 0.0f; // Prevent dragging into the floor
        DirectionToTarget.Normalize();

        float ForwardPushStrength = 2500.0f;
        float UpwardLiftStrength = 9800.0f; // Roughly counters half of real-world gravity for 20kg

        FVector FinalForce = (DirectionToTarget * ForwardPushStrength) + FVector(0.0f, 0.0f, UpwardLiftStrength);
        SpiderMesh->AddForce(FinalForce, NAME_None, false);
    }
}

void AONNX_Integration::RunSpiderInference()
{
    if (!ModelInstance.IsValid()) return;

    TConstArrayView<UE::NNE::FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
    int32 NumInputs = InputDescs.Num();

    if (NumInputs != 5) return; // Failsafe for mismatched ONNX versions

    TArray<UE::NNE::FTensorShape> InputShapes;
    TArray<UE::NNE::FTensorBindingCPU> InputBindings;
    InputShapes.SetNum(NumInputs);
    InputBindings.SetNumZeroed(NumInputs);

    // Map the buffers to the ONNX input tensors
    for (int32 i = 0; i < NumInputs; i++)
    {
        FString InputName = InputDescs[i].GetName();

        if (InputName.Contains(TEXT("obs")))
        {
            InputShapes[i] = UE::NNE::FTensorShape::Make({1, 132});
            InputBindings[i] = {ObservationsBuffer.GetData(), (uint64)(ObservationsBuffer.Num() * sizeof(float))};
        }
        else if (InputName.Contains(TEXT("height")))
        {
            InputShapes[i] = UE::NNE::FTensorShape::Make({1, 1, 64, 64});
            InputBindings[i] = {HeightmapBuffer.GetData(), (uint64)(HeightmapBuffer.Num() * sizeof(float))};
        }
        else if (InputName.Contains(TEXT("bev")))
        {
            InputShapes[i] = UE::NNE::FTensorShape::Make({1, 3, 64, 64});
            InputBindings[i] = {BevDataBuffer.GetData(), (uint64)(BevDataBuffer.Num() * sizeof(float))};
        }
        else if (InputName.Contains(TEXT("nav")))
        {
            InputShapes[i] = UE::NNE::FTensorShape::Make({1, 1, 33, 33});
            InputBindings[i] = {NavDataBuffer.GetData(), (uint64)(NavDataBuffer.Num() * sizeof(float))};
        }
        else if (InputName.Contains(TEXT("gru")) || InputName.Contains(TEXT("hidden")))
        {
            // Inject previous frame's memory
            InputShapes[i] = UE::NNE::FTensorShape::Make({1, 1, 256});
            InputBindings[i] = {GruPersistent.GetData(), (uint64)(GruPersistent.Num() * sizeof(float))};
        }
    }

    if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::EResultStatus::Ok) return;

    TConstArrayView<UE::NNE::FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();
    int32 NumOutputs = OutputDescs.Num();

    TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
    OutputBindings.SetNumZeroed(NumOutputs);

    for (int32 i = 0; i < NumOutputs; i++)
    {
        FString OutputName = OutputDescs[i].GetName();

        if (OutputName.Contains(TEXT("action")))
        {
            OutputBindings[i] = {ActionOutputBuffer.GetData(), (uint64)(ActionOutputBuffer.Num() * sizeof(float))};
        }
        else if (OutputName.Contains(TEXT("gru")) || OutputName.Contains(TEXT("hidden")))
        {
            OutputBindings[i] = {GruUpdated.GetData(), (uint64)(GruUpdated.Num() * sizeof(float))};
        }
    }

    // Execute the neural net
    if (ModelInstance->RunSync(InputBindings, OutputBindings) == UE::NNE::EResultStatus::Ok)
    {
        GruPersistent = GruUpdated; // Save state for next tick
    }
}

void AONNX_Integration::UpdateHeightMap()
{
    FVector RobotLocation = SpiderMesh->GetComponentLocation();
    FRotator RobotRotation = SpiderMesh->GetComponentRotation();
    
    // Only use Yaw so the scan grid stays flat with the horizon
    FRotator YawRotation(0.0f, RobotRotation.Yaw, 0.0f);

    int32 GridSize = 64;
    float GridSpacing = 10.0f; // 10cm spacing
    float HalfGrid = (GridSize / 2.0f) * GridSpacing;

    for (int32 x = 0; x < GridSize; x++)
    {
        for (int32 y = 0; y < GridSize; y++)
        {
            int32 BufferIndex = (x * GridSize) + y;

            float LocalX = (x * GridSpacing) - HalfGrid;
            float LocalY = (y * GridSpacing) - HalfGrid;
            FVector LocalPos(LocalX, LocalY, 0.0f);

            FVector WorldPos = YawRotation.RotateVector(LocalPos) + RobotLocation;

            FVector StartPos = WorldPos + FVector(0.0f, 0.0f, 200.0f);
            FVector EndPos = WorldPos + FVector(0.0f, 0.0f, -200.0f);

            FTraceDelegate TraceDelegate;
            TraceDelegate.BindLambda([this, BufferIndex, RobotLocation](const FTraceHandle& Handle, FTraceDatum& Data)
            {
                if (Data.OutHits.Num() > 0)
                {
                    // Isaac expects height relative to root, not absolute Z
                    float HitZ = Data.OutHits[0].ImpactPoint.Z;
                    float RelativeHeight = HitZ - RobotLocation.Z;
                    
                    if (HeightmapBuffer.IsValidIndex(BufferIndex))
                    {
                        HeightmapBuffer[BufferIndex] = RelativeHeight * 0.01f; 
                    }
                }
                else
                {
                    // Dropoff / bottomless pit fallback
                    if (HeightmapBuffer.IsValidIndex(BufferIndex))
                    {
                        HeightmapBuffer[BufferIndex] = -2.0f; 
                    }
                }
            });

            // Async trace to prevent locking up the main game thread
            GetWorld()->AsyncLineTraceByChannel(
                EAsyncTraceType::Single, StartPos, EndPos, ECC_Visibility, 
                FCollisionQueryParams::DefaultQueryParam, FCollisionResponseParams::DefaultResponseParam, &TraceDelegate);
        }
    }
}

void AONNX_Integration::UpdateBEVData(UTextureRenderTarget2D* RT)
{
    if (!RT) return;

    FTextureRenderTargetResource* RTResource = RT->GameThread_GetRenderTargetResource();
    TArray<FColor> RawPixels;

    if (!RTResource->ReadPixels(RawPixels)) return;

    const int32 NumPixels = 4096;
    BevDataBuffer.SetNumUninitialized(NumPixels * 3);

    // Unpack RGB into flat arrays for the tensor
    for (int32 i = 0; i < NumPixels; i++)
    {
        BevDataBuffer[i] = RawPixels[i].R / 255.0f;
        BevDataBuffer[i + NumPixels] = RawPixels[i].G / 255.0f;
        BevDataBuffer[i + (2 * NumPixels)] = RawPixels[i].B / 255.0f;
    }
}

void AONNX_Integration::UpdateObservations()
{
    if (ObservationsBuffer.Num() != 132) ObservationsBuffer.Init(0.0f, 132);

    // Ground contact check
    FHitResult GroundHit;
    FVector TraceStart = GetActorLocation() + FVector(0, 0, 50.0f);
    FVector TraceEnd = GetActorLocation() - FVector(0, 0, 150.0f);
    FCollisionQueryParams GroundParams(SCENE_QUERY_STAT(GroundTrace), true, this);
    bIsTouchingGround = GetWorld()->LineTraceSingleByChannel(GroundHit, TraceStart, TraceEnd, ECC_Visibility, GroundParams);

    UPrimitiveComponent* PhysicsBody = Cast<UPrimitiveComponent>(GetRootComponent());
    if (!PhysicsBody || !PhysicsBody->IsSimulatingPhysics())
    {
        PhysicsBody = FindComponentByClass<UPrimitiveComponent>();
    }

    APawn* PlayerPawn = UGameplayStatics::GetPlayerPawn(GetWorld(), 0);
    if (!PhysicsBody || !PlayerPawn) return;

    FTransform BodyTransform = PhysicsBody->GetComponentTransform();
    FVector MyLoc = BodyTransform.GetLocation();
    FVector TargetLoc = (TargetMode == ESpiderTargetMode::Player) ? PlayerPawn->GetActorLocation() : CurrentWaypoint;

    FVector RelativeVec = TargetLoc - MyLoc;
    FVector FlatRelativeVec = FVector(RelativeVec.X, RelativeVec.Y, 0.0f);
    float DistanceMeters = FlatRelativeVec.Size() * 0.01f;

    // Pathfinding logic update
    if (TargetMode == ESpiderTargetMode::Waypoint && DistanceMeters < 0.5f)
    {
        GenerateNewWaypoint();
        TargetLoc = CurrentWaypoint;
        RelativeVec = TargetLoc - MyLoc;
        FlatRelativeVec = FVector(RelativeVec.X, RelativeVec.Y, 0.0f);
        DistanceMeters = FlatRelativeVec.Size() * 0.01f;
    }

    // --- DEBUG VISUALS ---
    if (TargetMode == ESpiderTargetMode::Waypoint)
    {
        DrawDebugSphere(GetWorld(), TargetLoc, 20.0f, 12, FColor::Green, false, 0.1f);
    }
    else
    {
        // Draw a Red sphere on the player so you know it's hunting you
        DrawDebugSphere(GetWorld(), TargetLoc, 30.0f, 12, FColor::Red, false, 0.1f);
    }
    DrawDebugLine(GetWorld(), MyLoc, TargetLoc, FColor::Yellow, false, 0.1f);

    float ObsScale_LinVel = 2.0f; 
    float ObsScale_AngVel = 0.25f;
    float ObsScale_DofPos = 1.0f;  
    float ObsScale_DofVel = 0.05f; 

    FVector GlobalLinVel = PhysicsBody->GetPhysicsLinearVelocity();
    FVector LocalLinVel = BodyTransform.InverseTransformVectorNoScale(GlobalLinVel);
    
    // NOTE: Isaac Lab uses a different coordinate system, we must flip Y and invert targets!
    ObservationsBuffer[0] = (LocalLinVel.X * 0.01f) * ObsScale_LinVel;
    ObservationsBuffer[1] = (LocalLinVel.Y * 0.01f * -1.0f) * ObsScale_LinVel; 
    ObservationsBuffer[2] = (LocalLinVel.Z * 0.01f) * ObsScale_LinVel;

    FVector LocalAngVel = BodyTransform.InverseTransformVectorNoScale(PhysicsBody->GetPhysicsAngularVelocityInRadians());
    ObservationsBuffer[3] = LocalAngVel.X * ObsScale_AngVel;
    ObservationsBuffer[4] = (LocalAngVel.Y * -1.0f) * ObsScale_AngVel; 
    ObservationsBuffer[5] = LocalAngVel.Z * ObsScale_AngVel;

    FVector WorldGravity(0, 0, -1.0f);
    FVector LocalGravity = BodyTransform.InverseTransformVectorNoScale(WorldGravity);
    ObservationsBuffer[6] = LocalGravity.X;
    ObservationsBuffer[7] = LocalGravity.Y * -1.0f; 
    ObservationsBuffer[8] = LocalGravity.Z;

    FVector LocalTargetVec = BodyTransform.InverseTransformVectorNoScale(FlatRelativeVec).GetSafeNormal();
    ObservationsBuffer[9] = LocalTargetVec.X * -1.0f;
    ObservationsBuffer[10] = LocalTargetVec.Y * -1.0f;
    ObservationsBuffer[11] = LocalTargetVec.Z;
    ObservationsBuffer[12] = DistanceMeters;

    // Duplicate current target into next_target slot (no spline tracking yet)
    ObservationsBuffer[13] = LocalTargetVec.X * -1.0f;
    ObservationsBuffer[14] = LocalTargetVec.Y * -1.0f;
    ObservationsBuffer[15] = LocalTargetVec.Z;
    ObservationsBuffer[16] = DistanceMeters;

    ObservationsBuffer[17] = bIsTouchingGround ? 1.0f : 0.0f;

    float DeltaTime = FMath::Max(GetWorld()->GetDeltaSeconds(), 0.02f);

    for (int32 i = 0; i < 24; i++)
    {
        if (BoneNames.IsValidIndex(i))
        {
            int JointType = i % 4;
            float CurrentAngleDegrees = 0.0f;
            float DefaultAngle = 0.0f;

            FRotator LocalRot = SpiderMesh->GetSocketTransform(BoneNames[i], RTS_ParentBoneSpace).Rotator();

            if (JointType == 0) // HIP
            {
                CurrentAngleDegrees = LocalRot.Yaw;
                DefaultAngle = 0.0f;
            }
            else if (JointType == 1) // UPPER LEG
            {
                CurrentAngleDegrees = LocalRot.Roll;
                DefaultAngle = UpperDefault; 
            }
            else if (JointType == 2) // MIDDLE LEG
            {
                CurrentAngleDegrees = LocalRot.Roll;
                DefaultAngle = MidDefault; 
            }
            else if (JointType == 3) // LOWER LEG
            {
                CurrentAngleDegrees = LocalRot.Roll;
                DefaultAngle = LowerDefault; 
            }

            float RelativeAngleDeg = CurrentAngleDegrees - DefaultAngle;
            float RelativeAngleRad = FMath::DegreesToRadians(RelativeAngleDeg);

            // Joint Position
            ObservationsBuffer[18 + i] = RelativeAngleRad;

            // Joint Velocity
            float PrevAngleRad = PreviousJointAngles[i];
            float JointVelocityRad = (RelativeAngleRad - PrevAngleRad) / DeltaTime;
            ObservationsBuffer[42 + i] = JointVelocityRad;

            PreviousJointAngles[i] = RelativeAngleRad;
        }
    }

    // Pass previous actions into current observation
    for (int32 i = 0; i < 24; i++)
    {
        if (PreviousActions.IsValidIndex(i)) ObservationsBuffer[66 + i] = PreviousActions[i];
    }

    // Zero out unused state machine padding
    ObservationsBuffer[90] = 0.0f; // far_staleness
    ObservationsBuffer[91] = 0.0f; // can_see 
    for (int32 i = 92; i < 132; i++) ObservationsBuffer[i] = 0.0f;
    ObservationsBuffer[92] = 1.0f; // Flag state machine into Waypoint mode
}


void AONNX_Integration::ApplyActionsToConstraints()
{
    if (ControlNames.Num() != 24) return;

    const float ActionScale = 35.0f;

    for (int32 i = 0; i < 24; i++)
    {
        int JointType = i % 4;

        float RawAction = FMath::Clamp(ActionOutputBuffer[i], -1.0f, 1.0f);

        // Interpolate action to prevent violent motor snapping
        SmoothedActions[i] = FMath::Lerp(SmoothedActions[i], RawAction, 0.15f);
        float ScaledAction = SmoothedActions[i] * ActionScale;

        FPhysicsControlTarget Target;
        FRotator TargetRot = FRotator::ZeroRotator;

        // Apply calibrated resting pose offsets
        if (JointType == 0)      TargetRot.Yaw = ScaledAction;
        else if (JointType == 1) TargetRot.Roll = UpperDefault + ScaledAction;
        else if (JointType == 2) TargetRot.Roll = MidDefault + ScaledAction;
        else if (JointType == 3) TargetRot.Roll = LowerDefault + ScaledAction;

        Target.TargetOrientation = TargetRot;
        PhysicsControl->SetControlTarget(ControlNames[i], Target, true);
        PreviousActions[i] = RawAction;
    }

    if (SpiderMesh) SpiderMesh->WakeAllRigidBodies();
}

void AONNX_Integration::GenerateNewWaypoint()
{
    FVector SpiderLoc = GetActorLocation();

    // Generate random radial offset
    float RandomAngle = FMath::RandRange(0.0f, 360.0f);
    float RandomDistance = FMath::RandRange(200.0f, 500.0f);

    FVector Offset;
    Offset.X = FMath::Cos(FMath::DegreesToRadians(RandomAngle)) * RandomDistance;
    Offset.Y = FMath::Sin(FMath::DegreesToRadians(RandomAngle)) * RandomDistance;
    Offset.Z = 0.0f;

    FVector DraftWaypoint = SpiderLoc + Offset;

    // Trace down to snap waypoint to floor
    FHitResult Hit;
    FVector Start = DraftWaypoint + FVector(0, 0, 1000.0f); 
    FVector End = DraftWaypoint - FVector(0, 0, 1000.0f);   

    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this); 

    if (GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params))
    {
        CurrentWaypoint = Hit.ImpactPoint + FVector(0.0f, 0.0f, 25.0f);
    }
    else
    {
        CurrentWaypoint = DraftWaypoint; 
    }
}