package activationFunctions

import (
	"math"
	"neural-net/matrix"
)

// Sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid (for backpropagation)
func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// ReLU activation function
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Derivative of ReLU (for backpropagation)
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Tanh activation function
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// Derivative of Tanh (for backpropagation)
func TanhDerivative(x float64) float64 {
	t := Tanh(x)
	return 1 - t*t
}

// Leaky ReLU activation function
func LeakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

// Derivative of Leaky ReLU (for backpropagation)
func LeakyReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01
}

// Apply applies an activation function element-wise to a matrix
func Apply(sourceMatrix *matrix.Matrix, fn func(float64) float64) {
	for row := range sourceMatrix.Rows {
		for col := range row {
			sourceMatrix.Data[row][col] = fn(sourceMatrix.Data[row][col])
		}
	}
}

// ApplyNew creates a new matrix by applying an activation function element-wise
func ApplyNew(sourceMatrix *matrix.Matrix, fn func(float64) float64) *matrix.Matrix {
	result := matrix.NewMatrix(sourceMatrix.Rows, sourceMatrix.Cols)
	for row := range sourceMatrix.Rows {
		for col := range sourceMatrix.Cols {
			result.Data[row][col] = fn(sourceMatrix.Data[row][col])
		}
	}
	return result
}
