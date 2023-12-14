from Board import numbers, rows, columns, board
import numpy as np

# Find empty cells to fill in
def empty_cells(sudoku):
    return np.argwhere(sudoku == 0)

# Check to see if number is valid (compare with row, column and 3x3 box)
def valid_number(sudoku, number, row_number, column_number):
    # Get 3x3 box using math
    box_horizontal, box_vertical = (row_number) // 3 * 3, (column_number) // 3 * 3
    box = sudoku[(box_horizontal):(box_horizontal + 3), (box_vertical):(box_vertical + 3)]
    # Row/column/3x3 box
    return False if np.any(sudoku[row_number, :] == number) or np.any(sudoku[:, column_number] == number) or np.any(box == number) else True

# Global variable to store final result
result = None

# Solve board
def solve(puzzle, missing_cells=None, index=0):
    global result
    # Get missing cells
    if missing_cells is None:
        missing_cells = empty_cells(puzzle)
    # Board is solved
    if index >= len(missing_cells):
        result = np.copy(puzzle)
        return True
    # Get relevant missing cell
    else:
        row, column = missing_cells[index]
    # Recursively solve puzzle
    for number in range(1,10):
        if valid_number(puzzle, number, row, column):
            copy_sudoku = np.copy(puzzle)
            copy_sudoku[row,column] = number
            if solve(copy_sudoku, missing_cells, index+1) == True:
                return True
    return False

if __name__ == '__main__':
    sudoku = board(numbers(), rows(), columns())
    solve(sudoku)
    print(result)