#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"

#include "NNE.h"
#include "NNERuntimeCPU.h" // Or NNERuntimeGPU
#include "NNEModelData.h"
#include "NNETypes.h"
#include "Engine/TextureRenderTarget2D.h"
#include "TextureResource.h"
#include "Components/PrimitiveComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Components/PrimitiveComponent.h"
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
    class USkeletalMeshComponent *SpiderMesh;

    UPROPERTY()
    TArray<FName> BoneNames;

    UPROPERTY()
    TArray<float> PreviousActions;

    UPROPERTY()
    bool bIsTouchingGround = false;

    float GetJointAngle(UPhysicsConstraintComponent *Constraint);
    float GetJointVelocity(UPhysicsConstraintComponent *Constraint);
    void UpdateObservations();

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Physics")
    UPhysicsControlComponent *PhysicsControl;
    TArray<FName> ControlNames;

public:
    virtual void Tick(float DeltaTime) override;

    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    class UTextureRenderTarget2D *MyRenderTarget;

    UPROPERTY(EditAnywhere, Category = "Spider Bot")
    TObjectPtr<UNNEModelData> ModelData;

    struct InferenceTimerHandle;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI Navigation")
    FVector CurrentWaypoint;

    void GenerateNewWaypoint();

private:
    // Memory Buffers for LSTM (512 dimensions)
    TArray<float> HiddenState;
    TArray<float> CellState;
    const int32 LstmSize = 512;

    // These hold the "Current" memory
    TArray<float> h_Persistent;
    TArray<float> c_Persistent;

    // Temporary buffers to receive the "Next" memory from the model
    TArray<float> h_Updated;
    TArray<float> c_Updated;

    // Helper to run inference
    void RunSpiderInference();
    void UpdateHeightMap();

    void ApplyActionsToConstraints();
    void UpdateBEVData(UTextureRenderTarget2D *RT);
    void DrawLidarSensors(UPrimitiveComponent *PhysicsBody);

    TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

    // Buffers for I/O
    TArray<float> HeightmapBuffer;  // 4096 (64*64)
    TArray<float> HeightDataBuffer; // 4096 (64*64)
    TArray<float> BevDataBuffer;
    TArray<float> ActionOutputBuffer; // 24
    TArray<float> ObservationsBuffer;
    TArray<float> PreviousJointAngles;

    float BevTimer = 0.0f;
    FTimerHandle InferenceTimerHandle;

    UPROPERTY()
    float TimeSinceLastInference = 0.0f;

    const float InferenceInterval = 1.0f / 50.0f;
};