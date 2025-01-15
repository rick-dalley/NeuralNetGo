package main

import (
	"fmt"
	"neural-net/matrix"
	neuralNetwork "neural-net/neuralNetwork"
	"os"
	"strconv"
	"strings"
)

// reading the records one at a time
// func LoadMNIST(filename string) (*matrix.Matrix, []int) {
// 	file, err := os.Open(filename)
// 	if err != nil {
// 		panic(fmt.Sprintf("Error opening file: %v", err))
// 	}
// 	defer file.Close()

// 	reader := csv.NewReader(file)
// 	records, err := reader.ReadAll()
// 	if err != nil {
// 		panic(fmt.Sprintf("Error reading CSV: %v", err))
// 	}

// 	rows := len(records)
// 	cols := 784 // MNIST images are 28x28 flattened
// 	inputMatrix := matrix.NewMatrix(uint(rows), uint(cols))
// 	labels := make([]int, rows)

// 	for i, record := range records {
// 		// First value is the label
// 		label, _ := strconv.Atoi(record[0])
// 		labels[i] = label

// 		// Remaining values are the input data
// 		for j := 1; j <= cols; j++ {
// 			pixel, _ := strconv.Atoi(record[j])
// 			inputMatrix.Data[i][j-1] = float64(pixel) / 255.0 // Normalize
// 		}
// 	}

// 	return inputMatrix, labels
// }

// optimize version of LoadMNIST
func LoadMNIST(filename string) (*matrix.Matrix, []int) {
	data, err := os.ReadFile(filename)
	if err != nil {
		panic(fmt.Sprintf("Error reading file: %v", err))
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	rows := len(lines)
	cols := 784 // MNIST images are 28x28 flattened
	inputMatrix := matrix.NewMatrix(uint(rows), uint(cols))
	labels := make([]int, rows)

	for i, line := range lines {
		values := strings.Split(line, ",")

		// First value is the label
		label, _ := strconv.Atoi(values[0])
		labels[i] = label

		// Remaining values are the input data
		for j := 1; j <= cols; j++ {
			pixel, _ := strconv.Atoi(values[j])
			inputMatrix.Data[i][j-1] = float64(pixel) / 255.0 // Normalize
		}
	}

	return inputMatrix, labels
}

func main() {
	// Create a CPU profile file
	// cpuProfile, err := os.Create("cpu.prof")
	// if err != nil {
	// 	log.Fatal("could not create CPU profile: ", err)
	// }
	// defer cpuProfile.Close()

	// // Start CPU profiling
	// if err := pprof.StartCPUProfile(cpuProfile); err != nil {
	// 	log.Fatal("could not start CPU profile: ", err)
	// }
	// defer pprof.StopCPUProfile()

	// Load MNIST data
	inputMatrix, labels := LoadMNIST("mnist/mnist_train.csv")
	fmt.Println("Data loaded successfully")

	// Configure the neural network
	inputNodes := 784
	hiddenNodes := 100
	outputNodes := 10
	learningRate := 0.3
	epochs := uint(1)
	digits := uint(10)
	confidenceChanges := matrix.NewMatrix(epochs, digits)
	mnistRecords := inputMatrix.Rows
	// Initialize the neural network
	nn := neuralNetwork.NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
	fmt.Println("Neural network initialized")

	// Train the network
	fmt.Println("Training...")
	for epoch := uint(0); epoch < epochs; epoch++ {
		fmt.Printf("Epoch: (%d)\n Processing %d rows\n", epoch, mnistRecords)
		for i := uint(0); i < mnistRecords; i++ {
			// Extract input row as a slice
			inputRow := inputMatrix.Data[i]

			// Prepare target vector
			target := make([]float64, outputNodes)
			for j := range target {
				target[j] = 0.1 // Initialize to low confidence
			}
			target[labels[i]] = 0.99 // One-hot encode the correct label

			// Train the network
			nn.Train(inputRow, target)
			output := nn.ForwardPass(inputMatrix) // Query with the first input row
			if confidenceChanges.Data[epoch][labels[i]] < output.Data[0][labels[i]] {
				confidenceChanges.Data[epoch][labels[i]] = output.Data[0][labels[i]]
			}
			if i%1000 == 0 {
				fmt.Printf(".")
			}
		}
		fmt.Println()
		// Test the network
		fmt.Printf("\nresults:\n")
		fmt.Println(confidenceChanges.Data[epoch])
	}
	fmt.Println("Training completed")

}
