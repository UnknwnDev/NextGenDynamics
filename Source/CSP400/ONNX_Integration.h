// =========================================================================
// Project: Next Gen Dynamics
// DigiPen Institute of Technology
// Description: Handles Sim-to-Real transfer from Isaac Lab to UE5 using NNE. 
// Parses ONNX policy outputs into physical joint torques via Chaos Physics.
// =========================================================================

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"

#include "NNE.h"
#include "NNERuntimeCPU.h" 
#include "NNEModelData.h"
#include "NNETypes.h"
#include "Engine/TextureRenderTarget2D.h"
#include "TextureResource.h"
#include "Components/PrimitiveComponent.h"
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"
#include "TimerManager.h"
#include "PhysicsControlComponent.h"
#include "PhysicsControlData.h"

#include "ONNX_Integration.generated.h"

UENUM(BlueprintType)
enum class ESpiderTargetMode : uint8
{
    Player UMETA(DisplayName = "Follow Player"),
    Waypoint UMETA(DisplayName = "Roam to Waypoints")
};

UCLASS()
class CSP400_API AONNX_Integration : public AActor
{
    GENERATED_BODY()

public:
    AONNX_Integration();
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Navigation")
    ESpiderTargetMode TargetMode = ESpiderTargetMode::Waypoint;

protected:
    virtual void BeginPlay() override;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Spider Bot")
    class USkeletalMeshComponent* SpiderMesh;

    UPROPERTY()
    TArray<FName> BoneNames;

    UPROPERTY()
    TArray<float> PreviousActions;

    UPROPERTY()
    bool bIsTouchingGround = false;

    void UpdateObservations();

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Physics")
    UPhysicsControlComponent* PhysicsControl;
    TArray<FName> ControlNames;

public:
    virtual void Tick(float DeltaTime) override;

    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    class UTextureRenderTarget2D* MyRenderTarget;

    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    TObjectPtr<UNNEModelData> ModelData;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI Navigation")
    FVector CurrentWaypoint;

    void GenerateNewWaypoint();

private:
    // LSTM Memory Buffers
    TArray<float> HiddenState;
    TArray<float> CellState;
    const int32 LstmSize = 512;

    TArray<float> NavDataBuffer; 

    // GRU state tracking between inference frames
    TArray<float> GruPersistent;
    TArray<float> GruUpdated;

    void RunSpiderInference();
    void UpdateHeightMap();
    void ApplyActionsToConstraints();
    void UpdateBEVData(UTextureRenderTarget2D* RT);

    TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

    // I/O Buffers for the neural net
    TArray<float> HeightmapBuffer;  // 4096 (64x64 grid)
    TArray<float> HeightDataBuffer; 
    TArray<float> BevDataBuffer;
    TArray<float> ActionOutputBuffer; // 24 joints
    TArray<float> ObservationsBuffer;
    TArray<float> PreviousJointAngles;
    TArray<float> SmoothedActions;

    float BevTimer = 0.0f;
    FTimerHandle InferenceTimerHandle;

    UPROPERTY()
    float TimeSinceLastInference = 0.0f;

    // Locked to 50Hz to match Isaac Lab training
    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    float InferenceInterval = 1.0f / 50.0f;

    // Default resting poses
    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    float UpperDefault = 30.0f;

    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    float MidDefault = -60.0f;

    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    float LowerDefault = -30.0f;
};