package main

import (
	"encoding/csv"
	"fmt"
	"neural-net/activationFunctions"
	"neural-net/matrix"
	"os"
	"strconv"
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

func newNeuralNetwork(inputNodes int, hiddenNodes int, outputNodes int, learningRate float64) *NeuralNetwork {
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
func (nn *NeuralNetwork) train(input []float64, target []float64) {
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

func (nn *NeuralNetwork) forwardPass(inputs *matrix.Matrix) *matrix.Matrix {
	// Forward pass
	hiddenInputs := inputs.Multiply(&nn.InputHiddenWeights) // (N x I) x (I x H)
	hiddenOutputs := activationFunctions.ApplyNew(hiddenInputs, activationFunctions.Sigmoid)

	finalInputs := hiddenOutputs.Multiply(&nn.HiddenOutputWeights) // (N x H) x (H x O)
	finalOutputs := activationFunctions.ApplyNew(finalInputs, activationFunctions.Sigmoid)

	// Return the output as a Matrix
	return finalOutputs
}

func (nn *NeuralNetwork) query(input []float64) *matrix.Matrix {
	inputs := matrix.NewMatrix(1, uint(len(input)))
	copy(inputs.Data[0], input)
	return nn.forwardPass(inputs)
}

func LoadMNIST(filename string) (*matrix.Matrix, []int) {
	file, err := os.Open(filename)
	if err != nil {
		panic(fmt.Sprintf("Error opening file: %v", err))
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		panic(fmt.Sprintf("Error reading CSV: %v", err))
	}

	rows := len(records)
	cols := 784 // MNIST images are 28x28 flattened
	inputMatrix := matrix.NewMatrix(uint(rows), uint(cols))
	labels := make([]int, rows)

	for i, record := range records {
		// First value is the label
		label, _ := strconv.Atoi(record[0])
		labels[i] = label

		// Remaining values are the input data
		for j := 1; j <= cols; j++ {
			pixel, _ := strconv.Atoi(record[j])
			inputMatrix.Data[i][j-1] = float64(pixel) / 255.0 // Normalize
		}
	}

	return inputMatrix, labels
}

func main() {
	// Load MNIST data
	inputMatrix, labels := LoadMNIST("mnist/mnist_train.csv")
	fmt.Println("Data loaded successfully")

	// Configure the neural network
	inputNodes := 784
	hiddenNodes := 100
	outputNodes := 10
	learningRate := 0.3
	epochs := uint(10)
	digits := uint(10)
	confidenceChanges := matrix.NewMatrix(epochs, digits)
	// Initialize the neural network
	nn := newNeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
	fmt.Println("Neural network initialized")

	// Train the network
	fmt.Println("Training...")
	for epoch := uint(0); epoch < epochs; epoch++ {
		for i := uint(0); i < inputMatrix.Rows; i++ {
			// Extract input row as a slice
			inputRow := inputMatrix.Data[i]

			// Prepare target vector
			target := make([]float64, outputNodes)
			for j := range target {
				target[j] = 0.1 // Initialize to low confidence
			}
			target[labels[i]] = 0.99 // One-hot encode the correct label

			// Train the network
			nn.train(inputRow, target)
			// fmt.Printf(".")
			output := nn.forwardPass(inputMatrix) // Query with the first input row
			if confidenceChanges.Data[epoch][labels[i]] < output.Data[0][labels[i]] {
				confidenceChanges.Data[epoch][labels[i]] = output.Data[0][labels[i]]
			}
		}
		fmt.Println()
		// Test the network
		fmt.Println("Testing the network...")
		fmt.Println(confidenceChanges.Data[epoch])
	}
	fmt.Println("Training completed")

}
