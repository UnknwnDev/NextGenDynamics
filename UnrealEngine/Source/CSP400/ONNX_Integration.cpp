#include "ONNX_Integration.h"

AONNX_Integration::AONNX_Integration()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

void AONNX_Integration::BeginPlay()
{
	Super::BeginPlay();

	if (!ModelData)
	{
		UE_LOG(LogTemp, Error, TEXT("SPIDER: ModelData asset is NULL! Assign it in the Editor."));
		return;
	}

	// UE_LOG(LogTemp, Log, TEXT("DEBUG: Model Loaded"));
	PreviousActions.Init(0.0f, 24);
	ObservationsBuffer.Init(0.0f, 120);
	BevDataBuffer.Init(0.0f, 12288);
	HeightmapBuffer.Init(0.0f, 4096);
	PreviousJointAngles.Init(0.0f, 24);

	// Find the Skeletal Mesh attached to this Blueprint
	SpiderMesh = FindComponentByClass<USkeletalMeshComponent>();
	if (!SpiderMesh)
	{
		UE_LOG(LogTemp, Error, TEXT("SPIDER: No Skeletal Mesh found! Please add one to the Blueprint."));
		return;
	}

	// Ensure the physics engine is turned on for the rig
	SpiderMesh->SetSimulatePhysics(true);

	// Generate our exact 24 bone names
	BoneNames.Empty();
	ControlNames.Empty();

	for (int32 LegIdx = 0; LegIdx < 6; LegIdx++)
	{
		BoneNames.Add(FName(*FString::Printf(TEXT("hip_joint_%d"), LegIdx)));
		BoneNames.Add(FName(*FString::Printf(TEXT("upper_joint_%d"), LegIdx)));
		BoneNames.Add(FName(*FString::Printf(TEXT("middle_joint_%d"), LegIdx)));
		BoneNames.Add(FName(*FString::Printf(TEXT("lower_joint_%d"), LegIdx)));
	}

	// Setup Physics Control Component
	PhysicsControl = FindComponentByClass<UPhysicsControlComponent>();
	if (!PhysicsControl)
	{
		UE_LOG(LogTemp, Error, TEXT("SPIDER: No Physics Control Component found in Blueprint!"));
		return;
	}
	SpiderMesh->RecreatePhysicsState();

	// Create "Muscles" for all 24 bones
	for (FName BoneName : BoneNames)
	{
		FName ParentBone = SpiderMesh->GetParentBone(BoneName);

		if (ParentBone != NAME_None)
		{
			FPhysicsControlData ControlData;
			ControlData.LinearStrength = 0.0f;

			ControlData.AngularStrength = 50000000.0f;
			ControlData.AngularDampingRatio = 10.0f;

			FPhysicsControlTarget ControlTarget;

			FName MuscleName = PhysicsControl->CreateControl(SpiderMesh, ParentBone, SpiderMesh, BoneName, ControlData, ControlTarget, NAME_None, BoneName.ToString());

			ControlNames.Add(MuscleName);
		}
	}

	// Initialize all buffers with zeros
	h_Persistent.SetNumZeroed(LstmSize);
	c_Persistent.SetNumZeroed(LstmSize);
	ActionOutputBuffer.SetNum(24);
	h_Updated.SetNumZeroed(LstmSize);
	c_Updated.SetNumZeroed(LstmSize);

	TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));

	if (Runtime.IsValid())
	{
		TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);
		if (Model.IsValid())
		{
			ModelInstance = Model->CreateModelInstanceCPU();
			// UE_LOG(LogTemp, Log, TEXT("SPIDER: ONNX Model Instance created successfully!"));
		}
		else
		{
			// UE_LOG(LogTemp, Error, TEXT("SPIDER: Failed to create Model from ModelData."));
		}
	}
	else
	{
		// This is why you are crashing!
		// UE_LOG(LogTemp, Error, TEXT("SPIDER: NNE Runtime 'NNERuntimeORTCpu' not found. Check your Plugins!"));
	}

	TConstArrayView<UE::NNE::FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
	for (int32 i = 0; i < InputDescs.Num(); ++i)
	{
		auto Shape = InputDescs[i].GetShape();
		TConstArrayView<int32> Dims = Shape.GetData();

		uint64 TotalElements = 1;
		FString ShapeString = TEXT("");

		for (uint32 Dim : Dims)
		{
			TotalElements *= Dim;
			ShapeString += FString::Printf(TEXT("%u "), Dim);
		}

		// UE_LOG(LogTemp, Warning, TEXT("ONNX INPUT [%d]: Name: %s, Shape: [%s], Total Elements: %llu"),
		// 	   i, *InputDescs[i].GetName(), *ShapeString, TotalElements);
	}

	TConstArrayView<UE::NNE::FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();
	for (int32 i = 0; i < OutputDescs.Num(); ++i)
	{
		auto Shape = OutputDescs[i].GetShape();
		TConstArrayView<int32> Dims = Shape.GetData();

		uint64 TotalElements = 1;
		FString ShapeString = TEXT("");

		for (uint32 Dim : Dims)
		{
			TotalElements *= Dim;
			ShapeString += FString::Printf(TEXT("%u "), Dim);
		}

		// UE_LOG(LogTemp, Warning, TEXT("ONNX OUTPUT [%d]: Name: %s, Shape: [%s], Total Elements: %llu"),
		// 	   i, *OutputDescs[i].GetName(), *ShapeString, TotalElements);
	}
}

// Called every frame
void AONNX_Integration::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	TimeSinceLastInference += DeltaTime;

	if (TimeSinceLastInference >= InferenceInterval)
	{
		// 1. Update Map Data FIRST
		UpdateHeightMap();
		// UpdateBEVData(MyRenderTarget);

		// 2. Update Body Sensors
		UpdateObservations();

		// 3. Run the ONNX Model
		RunSpiderInference();
		SpiderMesh->WakeAllRigidBodies();

		// 4. Move the Muscles
		ApplyActionsToConstraints();

		TimeSinceLastInference -= InferenceInterval;
	}
}

void AONNX_Integration::RunSpiderInference()
{
	if (!ModelInstance.IsValid())
	{
		return;
	}

	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(UE::NNE::FTensorShape::Make({1, 120}));		  // 0: Observations
	InputShapes.Add(UE::NNE::FTensorShape::Make({1, 64, 64}));	  // 1: Heightmap
	InputShapes.Add(UE::NNE::FTensorShape::Make({1, 3, 64, 64})); // 2: BEV Data
	InputShapes.Add(UE::NNE::FTensorShape::Make({1, 1, 512}));	  // 3: LSTM h (Memory)
	InputShapes.Add(UE::NNE::FTensorShape::Make({1, 1, 512}));	  // 4: LSTM c (Memory)

	if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Error, TEXT("SPIDER: Failed to set ONNX input shapes!"));
		return;
	}

	TArray<UE::NNE::FTensorBindingCPU> InputBindings;
	InputBindings.SetNumZeroed(5);

	InputBindings[0] = {ObservationsBuffer.GetData(), (uint64)(ObservationsBuffer.Num() * sizeof(float))};
	InputBindings[1] = {HeightmapBuffer.GetData(), (uint64)(HeightmapBuffer.Num() * sizeof(float))};
	InputBindings[2] = {BevDataBuffer.GetData(), (uint64)(BevDataBuffer.Num() * sizeof(float))};
	InputBindings[3] = {h_Persistent.GetData(), (uint64)(h_Persistent.Num() * sizeof(float))};
	InputBindings[4] = {c_Persistent.GetData(), (uint64)(c_Persistent.Num() * sizeof(float))};

	TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
	OutputBindings.SetNumZeroed(3);

	OutputBindings[0] = {ActionOutputBuffer.GetData(), (uint64)(ActionOutputBuffer.Num() * sizeof(float))};
	OutputBindings[1] = {h_Updated.GetData(), (uint64)(h_Updated.Num() * sizeof(float))};
	OutputBindings[2] = {c_Updated.GetData(), (uint64)(c_Updated.Num() * sizeof(float))};

	if (ModelInstance->RunSync(InputBindings, OutputBindings) == UE::NNE::EResultStatus::Ok) // 0 means Success
	{
		h_Persistent = h_Updated;
		c_Persistent = c_Updated;
		// ApplyActionsToConstraints();
	}
}

void AONNX_Integration::UpdateHeightMap()
{
	const float GridSpacing = 10.0f;
	const float TraceHeight = 50.0f; // Start 50cm above
	const float TraceDepth = 150.0f; // Trace down

	FVector ActorLocation = GetActorLocation();
	if (UPrimitiveComponent *Root = Cast<UPrimitiveComponent>(GetRootComponent()))
	{
		ActorLocation = Root->GetComponentLocation();
	}

	// Clear buffer
	HeightmapBuffer.SetNumUninitialized(4096);

	FCollisionQueryParams TraceParams(SCENE_QUERY_STAT(HeightTrace), true, this);
	TraceParams.bReturnPhysicalMaterial = false;

	// Loop 64x64
	for (int32 y = 0; y < 64; y++)
	{
		for (int32 x = 0; x < 64; x++)
		{
			// Center the grid (0 to 63 -> -32 to +32)
			float OffsetX = (x - 32) * GridSpacing;
			float OffsetY = (y - 32) * GridSpacing;

			FVector Offset(OffsetX, OffsetY, 0);

			// To make the grid rotate with the spider (Local Space Lidar):
			Offset = GetActorQuat().RotateVector(Offset);

			FVector Start = ActorLocation + Offset + FVector(0, 0, TraceHeight);
			FVector End = Start - FVector(0, 0, TraceDepth);

			FHitResult Hit;
			float HeightValue = -1.0f; // Default "hole" value

			bool bHit = GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, TraceParams);

			if (bHit)
			{
				// Height relative to spider feet
				HeightValue = (Hit.ImpactPoint.Z - ActorLocation.Z) * 0.01f; // cm to meters

				// VISUALIZATION (Only draw every 4th point to save FPS)
				if (x % 4 == 0 && y % 4 == 0)
				{
					DrawDebugPoint(GetWorld(), Hit.ImpactPoint, 3.0f, FColor::Green, false, 0.05f);
				}
			}
			else
			{
				if (x % 4 == 0 && y % 4 == 0)
				{
					DrawDebugPoint(GetWorld(), End, 3.0f, FColor::Red, false, 0.05f);
				}
			}

			HeightmapBuffer[y * 64 + x] = HeightValue;
		}
	}
}

void AONNX_Integration::UpdateBEVData(UTextureRenderTarget2D *RT)
{
	if (!RT)
		return;

	FTextureRenderTargetResource *RTResource = RT->GameThread_GetRenderTargetResource();
	TArray<FColor> RawPixels;

	// 64x64 grid (4096 pixels)
	if (!RTResource->ReadPixels(RawPixels))
		return;

	// Ensure our buffer is exactly 3 * 4096 = 12288
	const int32 NumPixels = 4096;
	BevDataBuffer.SetNumUninitialized(NumPixels * 3);

	for (int32 i = 0; i < NumPixels; i++)
	{
		float R = RawPixels[i].R / 255.0f;
		float G = RawPixels[i].G / 255.0f;
		float B = RawPixels[i].B / 255.0f;

		// Index 0 to 4095
		BevDataBuffer[i] = R;
		// Index 4096 to 8191
		BevDataBuffer[i + NumPixels] = G;
		// Index 8192 to 12287
		BevDataBuffer[i + (2 * NumPixels)] = B;
	}
}

void AONNX_Integration::UpdateObservations()
{
	if (ObservationsBuffer.Num() != 120)
	{
		ObservationsBuffer.Init(0.0f, 120);
	}

	UPrimitiveComponent *PhysicsBody = Cast<UPrimitiveComponent>(GetRootComponent());
	if (!PhysicsBody || !PhysicsBody->IsSimulatingPhysics())
	{
		PhysicsBody = FindComponentByClass<UPrimitiveComponent>();
	}

	APawn *PlayerPawn = UGameplayStatics::GetPlayerPawn(GetWorld(), 0);

	if (!PhysicsBody || !PlayerPawn)
		return;

	FTransform BodyTransform = PhysicsBody->GetComponentTransform();
	FVector MyLoc = BodyTransform.GetLocation();

	FVector TargetLoc = CurrentWaypoint; // Default to waypoint

	// If set to Player, overwrite the target location!
	if (TargetMode == ESpiderTargetMode::Player)
	{
		TargetLoc = PlayerPawn->GetActorLocation();
	}

	// Get the raw directional vector to our chosen target
	FVector RelativeVec = TargetLoc - MyLoc;

	FVector FlatRelativeVec = FVector(RelativeVec.X, RelativeVec.Y, 0.0f);
	float DistanceMeters = FlatRelativeVec.Size() * 0.01f;

	// Only generate new waypoints if we are actually in Waypoint mode!
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

	// --- 2. root_lin_vel_b (Local Linear Velocity in m/s) ---
	FVector GlobalLinVel = PhysicsBody->GetPhysicsLinearVelocity();
	FVector LocalLinVel = BodyTransform.InverseTransformVectorNoScale(GlobalLinVel);
	ObservationsBuffer[0] = LocalLinVel.X * 0.01f;
	ObservationsBuffer[1] = LocalLinVel.Y * 0.01f * -1.0f;
	ObservationsBuffer[2] = LocalLinVel.Z * 0.01f;

	// --- 3. root_ang_vel_b (Local Angular Velocity in rad/s) ---
	FVector LocalAngVel = BodyTransform.InverseTransformVectorNoScale(PhysicsBody->GetPhysicsAngularVelocityInRadians());
	ObservationsBuffer[3] = LocalAngVel.X;
	ObservationsBuffer[4] = LocalAngVel.Y * -1.0f;
	ObservationsBuffer[5] = LocalAngVel.Z;

	// --- 4. projected_gravity_b ---
	FVector WorldGravity(0, 0, -1.0f);
	FVector LocalGravity = BodyTransform.InverseTransformVectorNoScale(WorldGravity);
	ObservationsBuffer[6] = LocalGravity.X;
	ObservationsBuffer[7] = LocalGravity.Y;
	ObservationsBuffer[8] = LocalGravity.Z;

	// --- 5. Target Info (Dynamically mapped!) ---
	FVector LocalTargetVec = BodyTransform.InverseTransformVectorNoScale(FlatRelativeVec).GetSafeNormal();

	ObservationsBuffer[9] = LocalTargetVec.X;
	ObservationsBuffer[10] = LocalTargetVec.Y * -1.0f;
	ObservationsBuffer[11] = LocalTargetVec.Z;
	ObservationsBuffer[12] = DistanceMeters;

	// --- 6. Next Target ---
	ObservationsBuffer[13] = LocalTargetVec.X;
	ObservationsBuffer[14] = LocalTargetVec.Y * -1.0f;
	ObservationsBuffer[15] = LocalTargetVec.Z;
	ObservationsBuffer[16] = DistanceMeters;

	// --- 17: is_contact ---
	ObservationsBuffer[17] = bIsTouchingGround ? 1.0f : 0.0f;

	// Get time since last frame to calculate velocity safely
	float DeltaTime = GetWorld()->GetDeltaSeconds();

	// --- 18-41: joint_pos AND 42-65: joint_vel ---
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
				DefaultAngle = 30.0f;
			}
			else if (JointType == 2) // MIDDLE LEG
			{
				CurrentAngleDegrees = LocalRot.Roll;
				DefaultAngle = -75.0f;
			}
			else if (JointType == 3) // LOWER LEG
			{
				CurrentAngleDegrees = LocalRot.Pitch;
				DefaultAngle = -45.0f;
			}

			int32 LegIdx = i / 4;

			float RelativeAngle = CurrentAngleDegrees - DefaultAngle;
			// if (bIsMirroredLeg)
			// 	RelativeAngle *= -1.0f;

			ObservationsBuffer[18 + i] = FMath::DegreesToRadians(RelativeAngle);
		}
	}

	// --- 66-89: Previous Actions ---
	for (int32 i = 0; i < 24; i++)
	{
		if (PreviousActions.IsValidIndex(i))
		{
			ObservationsBuffer[66 + i] = PreviousActions[i];
		}
	}

	// --- 90-97: Empty Padding ---
	for (int32 i = 90; i <= 97; i++)
	{
		ObservationsBuffer[i] = 0.0f;
	}

	// --- 98-100: Map Data & States (WAYPOINT ACTIVATED) ---
	ObservationsBuffer[98] = 1.0f;	// can_see player
	ObservationsBuffer[99] = 1.0f;	// one_hot_state: Waypoint (0)
	ObservationsBuffer[100] = 0.0f; // one_hot_state: Patrol (1)

	// Draw Debug Text
	if (PhysicsBody)
	{
		FString DistStr = FString::Printf(TEXT("Distance to Player: %.2f m"), ObservationsBuffer[12]);
		DrawDebugString(GetWorld(), MyLoc + FVector(0, 0, 150), DistStr, nullptr, FColor::Cyan, 0.05f, true, 1.5f);
	}

}

void AONNX_Integration::DrawLidarSensors(UPrimitiveComponent *PhysicsBody)
{
	if (!PhysicsBody)
		return;

	FVector ActorLocation = PhysicsBody->GetComponentLocation();
	FVector Forward = PhysicsBody->GetForwardVector();
	FVector Right = PhysicsBody->GetRightVector();
	FVector Up = PhysicsBody->GetUpVector();

	// Define the grid (e.g., a 5x5 grid around the spider)
	int32 GridSize = 30;
	float Spacing = 50.0f;	   // 50cm between points
	float TraceRange = 100.0f; // Look 2m down

	for (int32 x = -GridSize; x <= GridSize; x++)
	{
		for (int32 y = -GridSize; y <= GridSize; y++)
		{
			// Calculate point in a grid relative to spider's orientation
			FVector Offset = (Forward * x * Spacing) + (Right * y * Spacing);
			FVector Start = ActorLocation + Offset + Up; // Start slightly above
			FVector End = Start - Up;

			FHitResult Hit;
			FCollisionQueryParams Params;
			Params.AddIgnoredActor(this);

			bool bHit = GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params);

			FColor BeamColor = bHit ? FColor::Green : FColor::Red;

			// Draw a small point where the "Lidar" hits the ground
			if (bHit)
			{
				DrawDebugPoint(GetWorld(), Hit.ImpactPoint, 10.0f, BeamColor, false, 0.05f);
				// Draw the vertical "laser" beam
				DrawDebugLine(GetWorld(), Start, Hit.ImpactPoint + Up * TraceRange, BeamColor, false, 0.05f, 0, 1.0f);
			}
			else
			{
				DrawDebugLine(GetWorld(), Start, End, FColor::Red, false, 0.05f, 0, 1.0f);
			}
		}
	}
}

float AONNX_Integration::GetJointAngle(UPhysicsConstraintComponent *Constraint)
{
	if (!Constraint)
		return 0.0f;
	return FMath::DegreesToRadians(Constraint->GetCurrentSwing1());
}

float AONNX_Integration::GetJointVelocity(UPhysicsConstraintComponent *Constraint)
{
	if (!Constraint)
		return 0.0f;
	// UPrimitiveComponent* ChildComp = Cast<UPrimitiveComponent>(Constraint->OverrideComponent2.GetComponent(Constraint->GetOwner()));

	// // If OverrideComponent2 is null (common), we fallback to the standard ConstraintInstance lookup
	// if (!ChildComp)
	// {
	//     ChildComp = Cast<UPrimitiveComponent>(Constraint->ConstraintInstance.ConstraintComponent2.Get());
	// }

	// if (ChildComp && ChildComp->IsSimulatingPhysics())
	// {
	//     // Get velocity in Radians per second
	//     FVector AngVel = ChildComp->GetPhysicsAngularVelocityInRadians();

	//     // Return the axis the joint rotates on (X for most spider legs)
	//     return AngVel.X;
	// }
	return 0.0f;
}

void AONNX_Integration::ApplyActionsToConstraints()
{
	if (ControlNames.Num() != 24)
	{
		UE_LOG(LogTemp, Error, TEXT("CRITICAL FAIL: Only created %d muscles! Expected 24."), ControlNames.Num());
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 10.0f, FColor::Red, TEXT("MUSCLE CREATION FAILED. CHECK OUTPUT LOG."));
	}


	// The Isaac Lab action_scale (0.75 rad -> 42.97 deg)
	float ActionGain = 42.97f;

	for (int32 i = 0; i < 24; i++)
	{
		int LegIdx = i / 4;
		bool bIsMirroredLeg = (LegIdx == 0 || LegIdx == 4 || LegIdx == 5);
		// bool bIsMirroredLeg = false;

		float TargetAngleDelta = ActionOutputBuffer[i] * ActionGain;
		// float TargetAngleDelta = 0;

		// if (bIsMirroredLeg)
		// {
		// 	TargetAngleDelta *= -1.0f;
		// }

		int JointType = i % 4;
		FRotator TargetRot = FRotator::ZeroRotator;
		float FinalAngle = 0.0f;

		// if (JointType == 2)
		// {
		// 	// TEST 1: 30 degrees applied to PITCH (X)
		// 	TargetRot = FRotator(-90.0f, 00.0f, 00.0f);
		// }
		// else
		// {
		// 	TargetRot = FRotator(00.0f, -0.0f, 0.0f);
		// }

		if (JointType == 0) // HIP
		{
			FinalAngle = TargetAngleDelta;
			// FinalAngle = FMath::Clamp(FinalAngle, -45.0f, 45.0f);
			TargetRot = FRotator(0.0f, FinalAngle, -15);
		}
		else if (JointType == 1) // UPPER LEG
		{

			// if (bIsMirroredLeg)
			// 	FinalAngle = -TargetAngleDelta + 30.0f;
			// else
			// 	FinalAngle = TargetAngleDelta + -30.0f;
			FinalAngle = TargetAngleDelta  + 30;
			// FinalAngle = FMath::Clamp(FinalAngle, -10.0f, 60.0f);
			TargetRot = FRotator(0, 0.0f, FinalAngle);
		}
		else if (JointType == 2) // MIDDLE LEG
		{
			// if (bIsMirroredLeg)
			// 	FinalAngle = -TargetAngleDelta + -75.0f;
			// else
			// 	FinalAngle = TargetAngleDelta + -75.0f;
			FinalAngle = TargetAngleDelta - 75;
			// FinalAngle = FMath::Clamp(FinalAngle, -105.0f, -35.0f);
			TargetRot = FRotator(0, 0.0f, FinalAngle);
		}
		else if (JointType == 3) // LOWER LEG
		{

			// if (bIsMirroredLeg)
			// 	FinalAngle = -TargetAngleDelta + 45.0f;
			// else
			// 	FinalAngle = TargetAngleDelta + -45.0f;
			FinalAngle = TargetAngleDelta - 45;
			
			// FinalAngle = FMath::Clamp(FinalAngle, -85.0f, -5.0f);
			TargetRot = FRotator(0, 0.0f, FinalAngle);
		}

		FName MuscleName = ControlNames[i];

		// Build the Target Data
		FPhysicsControlTarget Target;
		Target.TargetOrientation = TargetRot;

		// Feed the new target to the muscle
		PhysicsControl->SetControlTarget(MuscleName, Target, true);

		// Save action for the next frame's observation
		PreviousActions[i] = ActionOutputBuffer[i];
	}
}

void AONNX_Integration::GenerateNewWaypoint()
{
	// 1. Get the spider's current location
	FVector SpiderLoc = GetActorLocation();

	// 2. Pick a random angle and a random distance (200cm to 500cm)
	float RandomAngle = FMath::RandRange(0.0f, 360.0f);
	float RandomDistance = FMath::RandRange(200.0f, 500.0f);

	// 3. Calculate the offset
	FVector Offset;
	Offset.X = FMath::Cos(FMath::DegreesToRadians(RandomAngle)) * RandomDistance;
	Offset.Y = FMath::Sin(FMath::DegreesToRadians(RandomAngle)) * RandomDistance;
	Offset.Z = 0.0f;

	FVector DraftWaypoint = SpiderLoc + Offset;

	// 4. Snap it to the floor using a Line Trace
	FHitResult Hit;
	FVector Start = DraftWaypoint + FVector(0, 0, 1000.0f); // Start 10m high
	FVector End = DraftWaypoint - FVector(0, 0, 1000.0f);	// Trace 10m deep

	FCollisionQueryParams Params;
	Params.AddIgnoredActor(this); // Don't hit the spider itself

	if (GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params))
	{
		CurrentWaypoint = Hit.ImpactPoint + FVector(0.0f, 0.0f, 25.0f);
	}
	else
	{
		CurrentWaypoint = DraftWaypoint; // Fallback if trace fails
	}
}