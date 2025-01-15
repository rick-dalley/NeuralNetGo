package neuralNetwork

import (
	"neural-net/activationFunctions"
	"neural-net/matrix"
)

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type NeuralNetwork struct {
	InputNodes          int
	OutputNodes         int
	HiddenNodes         int
	Epochs              int
	LearningRate        float64
	InputHiddenWeights  matrix.Matrix
	HiddenOutputWeights matrix.Matrix
}

func NewNeuralNetwork(inputNodes int, hiddenNodes int, outputNodes int, learningRate float64) *NeuralNetwork {
	nn := &NeuralNetwork{
		InputNodes:   inputNodes,
		HiddenNodes:  hiddenNodes,
		OutputNodes:  outputNodes,
		LearningRate: learningRate,
	}
	seed := uint64(12345)
	nn.InputHiddenWeights = *matrix.NewMatrix(uint(inputNodes), uint(hiddenNodes))
	nn.HiddenOutputWeights = *matrix.NewMatrix(uint(hiddenNodes), uint(outputNodes))

	nn.InputHiddenWeights.InitializeWeights(nn.InputNodes, seed)
	nn.HiddenOutputWeights.InitializeWeights(nn.HiddenNodes, seed)
	return nn
}
func (nn *NeuralNetwork) Train(input []float64, target []float64) {
	// Convert input slice to a 1xN matrix
	inputs := matrix.NewMatrix(1, uint(len(input)))
	copy(inputs.Data[0], input)

	// Convert target slice to a 1xN matrix
	targets := matrix.NewMatrix(1, uint(len(target)))
	copy(targets.Data[0], target)

	// Forward pass
	hiddenInputs := inputs.Multiply(&nn.InputHiddenWeights) // (1x784) x (784x100) = (1x100)
	hiddenOutputs := activationFunctions.ApplyNew(hiddenInputs, activationFunctions.Sigmoid)

	finalInputs := hiddenOutputs.Multiply(&nn.HiddenOutputWeights) // (1x100) x (100x10) = (1x10)
	finalOutputs := activationFunctions.ApplyNew(finalInputs, activationFunctions.Sigmoid)

	// Calculate output errors
	outputErrors := targets.Subtract(finalOutputs)
	hiddenErrors := outputErrors.Multiply(nn.HiddenOutputWeights.Transpose())

	// Update weights for hidden-to-output
	outputGradients := activationFunctions.ApplyNew(finalOutputs, func(x float64) float64 {
		return x * (1.0 - x)
	})

	scaledOutputErrors := outputErrors.ElementWiseMultiply(outputGradients)
	weightDeltaOutput := hiddenOutputs.Transpose().Multiply(scaledOutputErrors)
	weightDeltaOutput = weightDeltaOutput.MultiplyScalar(nn.LearningRate)
	nn.HiddenOutputWeights.Append(weightDeltaOutput)

	// Update weights for input-to-hidden
	hiddenGradients := activationFunctions.ApplyNew(hiddenOutputs, func(x float64) float64 {
		return x * (1.0 - x)
	})
	scaledHiddenErrors := hiddenErrors.ElementWiseMultiply(hiddenGradients)
	weightDeltaInput := inputs.Transpose().Multiply(scaledHiddenErrors)
	weightDeltaInput = weightDeltaInput.MultiplyScalar(nn.LearningRate)
	nn.InputHiddenWeights.Append(weightDeltaInput)
}

func (nn *NeuralNetwork) ForwardPass(inputs *matrix.Matrix) *matrix.Matrix {
	// Forward pass
	hiddenInputs := inputs.Multiply(&nn.InputHiddenWeights) // (N x I) x (I x H)
	hiddenOutputs := activationFunctions.ApplyNew(hiddenInputs, activationFunctions.Sigmoid)

	finalInputs := hiddenOutputs.Multiply(&nn.HiddenOutputWeights) // (N x H) x (H x O)
	finalOutputs := activationFunctions.ApplyNew(finalInputs, activationFunctions.Sigmoid)

	// Return the output as a Matrix
	return finalOutputs
}
