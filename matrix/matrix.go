package matrix

import (
	"fmt"
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Matrix struct {
	Rows uint
	Cols uint
	Data [][]float64
}

// NewMatrix creates a new generic matrix with the given rows and columns.
func NewMatrix(rows uint, cols uint) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// Copy creates a deep copy of the matrix.
func (m *Matrix) Copy() *Matrix {
	// Create a new matrix with the same dimensions
	copyMatrix := NewMatrix(m.Rows, m.Cols)

	// Copy each row of the matrix
	for i := uint(0); i < m.Rows; i++ {
		copy(copyMatrix.Data[i], m.Data[i])
	}

	return copyMatrix
}

// At sets a value at a specific position in the matrix.
func (m *Matrix) At(r uint, c uint, val float64) {
	m.Data[r][c] = val
}

func (m *Matrix) MultiplyScalar(multiplier float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for row := uint(0); row < result.Rows; row++ {
		for col := uint(0); col < result.Cols; col++ {
			result.Data[row][col] = m.Data[row][col] * multiplier
		}
	}
	return result
}

// func (m *Matrix) Multiply(other *Matrix) *Matrix {
// 	if m.Cols != other.Rows {
// 		panic(fmt.Sprintf("Matrix dimensions do not align for multiplication: m(%dx%d), other(%dx%d)",
// 			m.Rows, m.Cols, other.Rows, other.Cols))
// 	}

// 	result := NewMatrix(m.Rows, other.Cols)
// 	for i := uint(0); i < m.Rows; i++ {
// 		for j := uint(0); j < other.Cols; j++ {
// 			sum := 0.0
// 			for k := uint(0); k < m.Cols; k++ {
// 				sum += m.Data[i][k] * other.Data[k][j]
// 			}
// 			result.Data[i][j] = sum
// 		}
// 	}
// 	return result
// }

// Multiply - using worker pool
// func (m *Matrix) Multiply(other *Matrix) *Matrix {
// 	if m.Cols != other.Rows {
// 		panic(fmt.Sprintf("Matrix dimensions do not align for multiplication: m(%dx%d), other(%dx%d)",
// 			m.Rows, m.Cols, other.Rows, other.Cols))
// 	}

// 	result := NewMatrix(m.Rows, other.Cols)
// 	workerPool := make(chan struct{}, runtime.NumCPU()) // Limit concurrency to number of CPU cores
// 	var wg sync.WaitGroup

// 	for i := uint(0); i < m.Rows; i++ {
// 		for j := uint(0); j < other.Cols; j++ {
// 			wg.Add(1)
// 			workerPool <- struct{}{} // Acquire a slot in the pool
// 			go func(row, col uint) {
// 				defer wg.Done()
// 				defer func() { <-workerPool }() // Release the slot

// 				sum := 0.0
// 				for k := uint(0); k < m.Cols; k++ {
// 					sum += m.Data[row][k] * other.Data[k][col]
// 				}
// 				result.Data[row][col] = sum
// 			}(i, j)
// 		}
// 	}

// 	wg.Wait()
// 	return result
// }

func (m *Matrix) Multiply(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic(fmt.Sprintf("Matrix dimensions do not align for multiplication: m(%dx%d), other(%dx%d)",
			m.Rows, m.Cols, other.Rows, other.Cols))
	}

	// Convert `m` and `other` to gonum Dense matrices
	flatM := flatten(m.Data)
	flatOther := flatten(other.Data)
	matM := mat.NewDense(int(m.Rows), int(m.Cols), flatM)
	matOther := mat.NewDense(int(other.Rows), int(other.Cols), flatOther)

	// Perform multiplication using gonum
	matResult := mat.NewDense(int(m.Rows), int(other.Cols), nil)
	matResult.Mul(matM, matOther)

	// Convert the result back to our Matrix format
	result := &Matrix{
		Rows: uint(matResult.RawMatrix().Rows),
		Cols: uint(matResult.RawMatrix().Cols),
		Data: reshape(matResult.RawMatrix().Data, int(m.Rows), int(other.Cols)),
	}
	return result
}

// Helper function to flatten a 2D slice into a 1D slice
func flatten(data [][]float64) []float64 {
	flat := make([]float64, 0, len(data)*len(data[0]))
	for _, row := range data {
		flat = append(flat, row...)
	}
	return flat
}

// Helper function to reshape a 1D slice into a 2D slice
func reshape(data []float64, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for i := range result {
		result[i] = data[i*cols : (i+1)*cols]
	}
	return result
}

func (m *Matrix) ElementWiseMultiply(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions do not match for element-wise multiplication")
	}

	result := NewMatrix(m.Rows, m.Cols)
	for i := uint(0); i < m.Rows; i++ {
		for j := uint(0); j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * other.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows) // Note the flipped dimensions
	for row := uint(0); row < m.Rows; row++ {
		for col := uint(0); col < m.Cols; col++ {
			result.Data[col][row] = m.Data[row][col]
		}
	}
	return result
}

func (m *Matrix) Append(other *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions do not match for addition")
	}

	for row := uint(0); row < m.Rows; row++ {
		for col := uint(0); col < m.Cols; col++ {
			m.Data[row][col] += other.Data[row][col]
		}
	}
}

func (m *Matrix) Subtract(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions do not match for subtraction")
	}

	result := NewMatrix(m.Rows, m.Cols)
	for row := uint(0); row < m.Rows; row++ {
		for col := uint(0); col < m.Cols; col++ {
			result.Data[row][col] = m.Data[row][col] - other.Data[row][col]
		}
	}
	return result
}

// Sigmoid applies the sigmoid function element-wise (only for float64).
func (m *Matrix) Sigmoid() {
	for i := uint(0); i < m.Rows; i++ {
		for j := uint(0); j < m.Cols; j++ {
			val := float64(m.Data[i][j])
			m.Data[i][j] = float64(1.0 / (1.0 + math.Exp(-val)))
		}
	}
}

// SigmoidPrime applies the derivative of the sigmoid function element-wise (only for float64).
func (m *Matrix) SigmoidPrime() {
	for i := uint(0); i < m.Rows; i++ {
		for j := uint(0); j < m.Cols; j++ {
			sigmoid := 1.0 / (1.0 + math.Exp(-m.Data[i][j]))
			m.Data[i][j] = sigmoid * (1.0 - sigmoid)
		}
	}
}

func (m *Matrix) InitializeWeights(nodesInPreviousLayer int, seed uint64) {
	if m.Data == nil {
		panic("Matrix must be initialized before calling InitializeWeights")
	}

	// Seed the random number generator
	rng := rand.New(rand.NewSource(seed))

	// Parameters for the normal distribution
	mean := 0.0
	stddev := 1.0 / math.Sqrt(float64(nodesInPreviousLayer)) // Xavier initialization

	// Create a normal distribution
	normalDist := distuv.Normal{
		Mu:    mean,
		Sigma: stddev,
		Src:   rng,
	}

	// Populate the matrix with random weights
	for row := uint(0); row < m.Rows; row++ {
		for col := uint(0); col < m.Cols; col++ {
			m.Data[row][col] = normalDist.Rand()
		}
	}
}
