package nn;
import "core:math"
import "core:math/rand"
import "core:fmt"
import "core:io"

numInput :: 2;
numHiddenNodes :: 2;
numOutputs :: 1;
numTrainingSets :: 4;

init_weight :: proc()->f64{
    return rand.float64_range(0,1)
}

sigmoid :: proc(x:f64)->f64{
    return 1/(1+math.exp(-x))
}
dsigmoid :: proc(x:f64)->f64{
    return x * (1-x)
}


main :: proc() {
    lr :f64 = 0.05;
    hiddenLayer :[numHiddenNodes]f64;
    outputLayer :[numOutputs]f64;

    hiddenLayerBias:[numHiddenNodes]f64;
    outputLayerBias:[numOutputs]f64;

    hiddenWeights:[numInput][numHiddenNodes]f64;
    outputWeights:[numHiddenNodes][numOutputs]f64;

    training_inputs :[numTrainingSets][numInput]f64={{0.0,0.0},{1.0,0.0},{0.0,1.0},{1.0,1.0}};
    training_outputs :[numTrainingSets][numOutputs]f64={{0.0},{1.0},{1.0},{0.0}};
    
    for i:= 0; i < numInput; i+=1 {
        for j:=0; j <numHiddenNodes; j+=1 {
            hiddenWeights[i][j] = init_weight();
        }
    } 

    for i:= 0; i < numHiddenNodes; i+=1 {
        for j:=0; j <numOutputs; j+=1 {
            outputWeights[i][j] = init_weight();
        }
    } 

    for i:= 0; i <numOutputs; i+=1 {
        outputLayerBias[i] = init_weight();
    }


    trainingSetOrder :[]int = {0,1,2,3}


    numberOfEpochs := 100000;

    //Training loop
    for epoch:int=0; epoch <numberOfEpochs; epoch +=1 {
        rand.shuffle(trainingSetOrder)
        for x:=0; x <numTrainingSets; x +=1 {
            i:= trainingSetOrder[x]

            //forward pass

            //Hidden layer activation
            for j:=0; j < numHiddenNodes; j+=1 {
                activation := hiddenLayerBias[j]
                
                for k:=0; k < numInput; k+=1 {
                    activation += training_inputs[i][k] * hiddenWeights[k][j]
                }

                hiddenLayer[j] = sigmoid(activation);
            }
            //Output Activation
            for j:=0; j < numOutputs; j+=1 {
                activation := outputLayerBias[j]
                
                for k:=0; k < numHiddenNodes; k+=1 {
                    activation += hiddenLayer[k] * outputWeights[k][j]
                }

                outputLayer[j] = sigmoid(activation);
            }
            //fmt.printf("Forward Pass:\n Input: %g %g \n Output: %g \n Expected Output: %g \n",training_inputs[i][0],training_inputs[i][1],outputLayer[0],
            //training_outputs[i][0]);
            if(epoch%%1000 == 0 ){
            fmt.printf("%g \n", outputLayer[0] - training_outputs[i][0])
        }
            //Backprop
            
            //Change in output

            deltaOutput : [numOutputs]f64;
            for j:=0; j < numOutputs; j+=1 {
                error: f64 = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dsigmoid(outputLayer[j])
            }

            //compute change in hidden weight
            deltaHidden :[numHiddenNodes]f64;
            for j:=0; j < numHiddenNodes; j+=1 {
                error: f64 = 0.0;
                for k:= 0; k<numOutputs; k+=1 {
                    error += deltaOutput[k]* outputWeights[j][k];
                }
                deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
            }
            // Apply the changes in output weights
            for j:=0; j < numOutputs; j+=1 {
                outputLayerBias[j] += deltaOutput[j] * lr
                for k:=0; k< numHiddenNodes; k+=1 {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr
                }
            }
            // Apply the changes in hidden weights
            for j:=0; j < numHiddenNodes; j+=1 {
                hiddenLayerBias[j] += deltaHidden[j] * lr
                for k:=0; k< numInput; k+=1 {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr
                }
            }

        }
        
    }

}
