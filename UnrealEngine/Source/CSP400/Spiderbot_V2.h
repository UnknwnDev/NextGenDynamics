// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"

#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h" // Or NNERuntimeGPU 
#include "Engine/Texture.h"

#include "Spiderbot_V2.generated.h"

UCLASS()
class CSP400_API ASpiderbot_V2 : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ASpiderbot_V2();
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spider Properties")
    float str = 50000.0;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spider Properties")
    float damp = 4000.0;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spider Properties")
    float hip = -5.0;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spider Properties")
    float upper = 45.0;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spider Properties")
    float middle = 75.0;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spider Properties")
    float lower = 45.0;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;


    #if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
    #endif

    // Cached arrays so we don't have to search every frame
    TArray<UPhysicsConstraintComponent*> HipJoints;
    TArray<UPhysicsConstraintComponent*> UpperJoints;
    TArray<UPhysicsConstraintComponent*> MiddleJoints;
    TArray<UPhysicsConstraintComponent*> LowerJoints;

    TArray<UPhysicsConstraintComponent*> AllJoints;

    UPROPERTY(EditAnywhere, Category = "Spider|AI")
    UNNEModelData* ModelData;

    TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

    // We store the observation and action buffers
    TArray<float> Observations;
    TArray<float> HeightData;
    TArray<float> Actions;

    void RunModelInference();

    void CollectObservations();
    void CollectHeightData();

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

    void TriggerPeriodicAction();

    FTimerHandle ActionTimerHandle;
    
    // Helper to apply settings to a specific list
    void ApplyToJointList(TArray<UPhysicsConstraintComponent*>& List, float Pitch, float Strength, float Damping);
    
    /**
     * @param Strength - Position Strength
     * @param Damping - Velocity Strength
     * @param HipPitch - The target pitch value
     * @param UpperPitch - The target pitch value
     * @param MiddlePitch - The target pitch value
     * @param LowerPitch - The target pitch value
     */
    UFUNCTION(BlueprintCallable, Category = "Spider Physics")
    void SetAction(float Strength, float Damping, float HipPitch, float UpperPitch, float MiddlePitch, float LowerPitch);
    void ApplyActions();

private:
    TArray<float> HiddenState;
    TArray<float> CellState;

    const int32 LstmDim = 512;
};
