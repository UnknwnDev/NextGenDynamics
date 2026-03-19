// Fill out your copyright notice in the Description page of Project Settings.

#include "Spiderbot_V2.h"

// Sets default values
ASpiderbot_V2::ASpiderbot_V2()
{
    // Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void ASpiderbot_V2::BeginPlay()
{
    Super::BeginPlay();

    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("SPIDERBOT: ModelData is MISSING! Assign it in the Editor."));
        return;
    }

    // TArray<FString> RuntimeNames = UE::NNE::GetAllRuntimeNames<INNERuntimeCPU>();
    // UE_LOG(LogTemp, Warning, TEXT("--- AVAILABLE NNE RUNTIMES ---"));
    // for (const FString& Name : RuntimeNames)
    // {
    //     UE_LOG(LogTemp, Warning, TEXT("Found Runtime: %s"), *Name);
    // }
    // UE_LOG(LogTemp, Warning, TEXT("------------------------------"));

    HiddenState.SetNumZeroed(LstmDim);
    CellState.SetNumZeroed(LstmDim);

    TArray<UPhysicsConstraintComponent *> AllConstraints;
    GetComponents<UPhysicsConstraintComponent>(AllConstraints);

    for (UPhysicsConstraintComponent *Joint : AllConstraints)
    {
        FString Name = Joint->GetName();
        if (Name.Contains(TEXT("hip_joint")))
            HipJoints.Add(Joint);
        if (Name.Contains(TEXT("upper_joint")))
            UpperJoints.Add(Joint);
        if (Name.Contains(TEXT("middle_joint")))
            MiddleJoints.Add(Joint);
        if (Name.Contains(TEXT("lower_joint")))
            LowerJoints.Add(Joint);
    }

    AllJoints.Empty();
    AllJoints.Append(HipJoints);
    AllJoints.Append(UpperJoints);
    AllJoints.Append(MiddleJoints);
    AllJoints.Append(LowerJoints);

    // // Set the timer to run every 5.0 seconds, looping (true)
    // GetWorldTimerManager().SetTimer(
    //     ActionTimerHandle,
    //     this,
    //     &ASpiderbot_V2::TriggerPeriodicAction,
    //     5.0f,
    //     true);

    SetAction(str, damp, hip, upper, middle, lower);

    if (ModelData)
    {
        GEngine->AddOnScreenDebugMessage(
            -1,                                   // Key to prevent duplicate messages (-1 means no key)
            2.0f,                                 // Duration (seconds)
            FColor::Yellow,                       // Text color
            FString::Printf(TEXT("Model Found.")) // Formatted message
        );
        // Get the NNE Runtime (CPU is easiest to start with)
        // TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
        // if (Runtime.IsValid())
        // {
        //     GEngine->AddOnScreenDebugMessage(
        //         -1,                                     // Key to prevent duplicate messages (-1 means no key)
        //         2.0f,                                   // Duration (seconds)
        //         FColor::Yellow,                         // Text color
        //         FString::Printf(TEXT("Runtime Valid.")) // Formatted message
        //     );
        //     TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);

        //     if (Model.IsValid())
        //     {
        //         // 2. Create the Model Instance from the Model
        //         ModelInstance = Model->CreateModelInstanceCPU();
        //         TArray<UE::NNE::FTensorShape> InputShapes;

        //         Observations.SetNumZeroed(84);
        //         HeightData.SetNumZeroed(121);
        //         HiddenState.SetNumZeroed(512);
        //         CellState.SetNumZeroed(512);
        //         Actions.SetNumZeroed(24); 

        //         // Add shapes in the EXACT order shown in the Model Data asset window
        //         InputShapes.Add(UE::NNE::FTensorShape::Make({1, 84}));     // observations
        //         InputShapes.Add(UE::NNE::FTensorShape::Make({1, 121}));    // height_data
        //         InputShapes.Add(UE::NNE::FTensorShape::Make({1, 1, 512})); // hidden_state
        //         InputShapes.Add(UE::NNE::FTensorShape::Make({1, 1, 512})); // cell_state

        //         // 3. Tell the model to prepare for these shapes
        //         if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::EResultStatus::Ok)
        //         {
        //             UE_LOG(LogTemp, Error, TEXT("Failed to set input tensor shapes!"));
        //         }
        //         else
        //         {
        //             UE_LOG(LogTemp, Log, TEXT("Tensor shapes set successfully."));
        //         }
        //         GEngine->AddOnScreenDebugMessage(
        //             -1,                                   // Key to prevent duplicate messages (-1 means no key)
        //             2.0f,                                 // Duration (seconds)
        //             FColor::Yellow,                       // Text color
        //             FString::Printf(TEXT("Model Valid.")) // Formatted message
        //         );
        //     }
        //     else
        //     {
        //         UE_LOG(LogTemp, Error, TEXT("Failed to create NNE Model from data."));
        //     }
        // }
    }
}

// Called every frame
void ASpiderbot_V2::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (!ModelInstance.IsValid())
    {
        GEngine->AddOnScreenDebugMessage(
            -1,                                                // Key to prevent duplicate messages (-1 means no key)
            2.0f,                                              // Duration (seconds)
            FColor::Yellow,                                    // Text color
            FString::Printf(TEXT("Model Instance not loaded")) // Formatted message
        );
        return;
    }

    CollectObservations();
    CollectHeightData();

    // Bind multiple inputs
    TArray<UE::NNE::FTensorBindingCPU> InputBindings;
    InputBindings.Add({Observations.GetData(), (uint64)Observations.Num() * sizeof(float)});
    InputBindings.Add({HeightData.GetData(), (uint64)HeightData.Num() * sizeof(float)});

    TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
    OutputBindings.Add({Actions.GetData(), (uint64)(Actions.Num() * sizeof(float))});

    if (ModelInstance->RunSync(InputBindings, OutputBindings) == UE::NNE::EResultStatus::Ok)
    {
        GEngine->AddOnScreenDebugMessage(
            -1,                                     // Key to prevent duplicate messages (-1 means no key)
            2.0f,                                   // Duration (seconds)
            FColor::Yellow,                         // Text color
            FString::Printf(TEXT("Appying Action")) // Formatted message
        );
        ApplyActions();
    }

    // RunModelInference();
}

void ASpiderbot_V2::RunModelInference()
{
    TArray<UE::NNE::FTensorBindingCPU> InputBindings;
    InputBindings.Add({Observations.GetData(), (uint64)Observations.Num() * sizeof(float)});
    InputBindings.Add({HeightData.GetData(), (uint64)HeightData.Num() * sizeof(float)});
    InputBindings.Add({HiddenState.GetData(), (uint64)HiddenState.Num() * sizeof(float)}); // MEMORY IN
    InputBindings.Add({CellState.GetData(), (uint64)CellState.Num() * sizeof(float)});     // MEMORY IN

    // Most models output [Actions, NewHiddenState, NewCellState]
    TArray<float> NewHidden;
    NewHidden.SetNumZeroed(LstmDim);
    TArray<float> NewCell;
    NewCell.SetNumZeroed(LstmDim);

    TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
    OutputBindings.Add({Actions.GetData(), (uint64)Actions.Num() * sizeof(float)});
    OutputBindings.Add({NewHidden.GetData(), (uint64)LstmDim * sizeof(float)}); // MEMORY OUT
    OutputBindings.Add({NewCell.GetData(), (uint64)LstmDim * sizeof(float)});   // MEMORY OUT

    if (ModelInstance->RunSync(InputBindings, OutputBindings) == UE::NNE::EResultStatus::Ok)
    {
        // 3. Update memory for the NEXT tick
        HiddenState = NewHidden;
        CellState = NewCell;

        ApplyActions();
    }
}

void ASpiderbot_V2::CollectObservations()
{
    Observations.Empty(); // Clear for fresh data

    // Linear/Angular Velocity (Local Space)
    FVector LinVel = GetActorRotation().UnrotateVector(GetVelocity()) * 0.1f; // Scaled
    Observations.Add(LinVel.X);
    Observations.Add(LinVel.Y);
    Observations.Add(LinVel.Z);

    if (UPrimitiveComponent *RootPrim = Cast<UPrimitiveComponent>(GetRootComponent()))
    {
        FVector AngVel = GetActorRotation().UnrotateVector(RootPrim->GetPhysicsAngularVelocityInDegrees()) * 0.5f;
        Observations.Add(AngVel.X);
        Observations.Add(AngVel.Y);
        Observations.Add(AngVel.Z);
    }
    else
    {
        // Fallback if root isn't physics-enabled
        Observations.Add(0.f);
        Observations.Add(0.f);
        Observations.Add(0.f);
    }

    // Projected Gravity (Direction of 'Down' relative to the robot)
    FVector Gravity = GetActorRotation().UnrotateVector(FVector(0, 0, -1));
    Observations.Add(Gravity.X);
    Observations.Add(Gravity.Y);
    Observations.Add(Gravity.Z);

    // Joint Positions (Normalized -1 to 1)
    for (UPhysicsConstraintComponent *Joint : AllJoints)
    {
        float CurrentAngle = Joint->GetCurrentSwing1();
        Observations.Add(CurrentAngle / 45.0f);
    }
}

void ASpiderbot_V2::CollectHeightData()
{
    HeightData.Empty();
    FVector BaseLoc = GetActorLocation();

    for (float x = -50; x <= 50; x += 10)
    {
        for (float y = -50; y <= 50; y += 10)
        {
            FVector TraceStart = BaseLoc + GetActorRotation().RotateVector(FVector(x, y, 100));
            FVector TraceEnd = TraceStart - FVector(0, 0, 200);

            FHitResult Hit;
            if (GetWorld()->LineTraceSingleByChannel(Hit, TraceStart, TraceEnd, ECC_Visibility))
            {
                HeightData.Add((Hit.Location.Z - BaseLoc.Z) * 0.1f); // Relative height
            }
            else
            {
                HeightData.Add(-1.0f);
            }
        }
    }
}

#if WITH_EDITOR
void ASpiderbot_V2::PostEditChangeProperty(FPropertyChangedEvent &PropertyChangedEvent)
{
    SetAction(str, damp, hip, upper, middle, lower);

    Super::PostEditChangeProperty(PropertyChangedEvent);
}
#endif

void ASpiderbot_V2::TriggerPeriodicAction()
{
    static bool bToggle = false;
    bToggle = !bToggle;

    if (bToggle)
        SetAction(str, damp, 0, upper - 20, middle + 40, lower + 40);
    else
        SetAction(str, damp, hip, upper, middle, lower);

    GEngine->AddOnScreenDebugMessage(-1, 2.0f, FColor::Cyan, TEXT("Periodic Jump!"));
}

void ASpiderbot_V2::ApplyToJointList(TArray<UPhysicsConstraintComponent *> &List, float Pitch, float Strength, float Damping)
{
    FRotator TargetRotation = FRotator(Pitch, 0.0f, 0.0f);
    FVector TargetVelocity = FVector::ZeroVector;

    for (UPhysicsConstraintComponent *Joint : List)
    {
        Joint->SetAngularVelocityTarget(TargetVelocity);
        Joint->SetAngularOrientationTarget(TargetRotation);
        Joint->SetAngularDriveParams(Strength, Damping, 0.0f);
        Joint->SetOrientationDriveTwistAndSwing(true, true);
        Joint->SetAngularDriveMode(EAngularDriveMode::TwistAndSwing);

        // Wake up the physics bodies so they don't ignore the command
        if (UPrimitiveComponent *PhysParent = Cast<UPrimitiveComponent>(Joint->GetOwner()->GetRootComponent()))
        {
            PhysParent->WakeAllRigidBodies();
        }
    }
}

void ASpiderbot_V2::SetAction(float Strength, float Damping, float HipPitch, float UpperPitch, float MiddlePitch, float LowerPitch)
{
    ApplyToJointList(HipJoints, HipPitch, Strength, Damping);
    ApplyToJointList(UpperJoints, UpperPitch, Strength, Damping);
    ApplyToJointList(MiddleJoints, MiddlePitch, Strength, Damping);
    ApplyToJointList(LowerJoints, LowerPitch, Strength, Damping);
}

void ASpiderbot_V2::ApplyActions()
{
    // The scale determines how 'aggressive' the movement is.
    // Start small (20.0) and increase to match the Isaac Sim training.
    float ActionScale = 35.0f;
    if (Actions[0])
    {
        GEngine->AddOnScreenDebugMessage(
            -1,                                                 // Key to prevent duplicate messages (-1 means no key)
            2.0f,                                               // Duration (seconds)
            FColor::Yellow,                                     // Text color
            FString::Printf(TEXT("Actions[0]: %f"), Actions[0]) // Formatted message
        );
    }

    // Actions[0-5] = Hips, [6-11] = Upper, etc. (Depending on your training order)
    SetAction(
        str,
        damp,
        Actions[0] * ActionScale,
        Actions[6] * ActionScale,
        Actions[12] * ActionScale,
        Actions[18] * ActionScale);

    GEngine->AddOnScreenDebugMessage(
        -1,                                   // Key to prevent duplicate messages (-1 means no key)
        2.0f,                                 // Duration (seconds)
        FColor::Yellow,                       // Text color
        FString::Printf(TEXT("Apply Action")) // Formatted message
    );
}