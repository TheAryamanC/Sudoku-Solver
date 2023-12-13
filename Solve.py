from Board import numbers, rows, columns, board
import numpy as np
import math

# Find empty cells to fill in
def empty_cells(sudoku):
    missing_cells = np.where(sudoku == 0)
    return list(zip(missing_cells[0], missing_cells[1]))

# Check to see if number is valid (compare with row, column and 3x3 box)
def valid_number(sudoku, number, row_number, column_number):
    # Row
    row = sudoku[row_number-1][0]
    if number in row:
        return False
    # Column
    column = [i.tolist()[0][0] for i in sudoku[:, column_number-1]]
    if number in column:
        return False
    # Box
    box_horizontal = math.floor((row_number - 1) / 3) + 1
    box_vertical = math.floor((column_number - 1) / 3) + 1
    box = np.take(sudoku, [3*box_horizontal-3,3*box_horizontal-2,3*box_horizontal-1], axis=1)[3*box_vertical - 3: 3*box_vertical]
    box = [i.tolist()[0] for i in box]
    if number in box[0] or number in box[1] or number in box[2]:
        return False
    # None - number is possible
    return True

# Solve board
def solve(puzzle, missing_cells=None, index=0):
    # Get missing cells
    if missing_cells == None:
        missing_cells = empty_cells(puzzle)
    # Board is solved
    if index > len(missing_cells):
        return True
    # Get relevant missing cell
    else:
        row = missing_cells[index][0] + 1
        columns = missing_cells[index][1] + 1
    # Recursively solve puzzle
    for number in range(1,10):
        if valid_number(puzzle, number, row, columns) == False:
            continue
        else:
            copy_sudoku = puzzle
            copy_sudoku[row-1,columns-1] = number
            if solve(copy_sudoku, missing_cells, index+1) == True:
                return True
    return False

if __name__ == '__main__':
    puzzle, rws, cols = numbers(), rows(), columns()
    sudoku = board(puzzle, rws, cols)
    print(solve(sudoku))