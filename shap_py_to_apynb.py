import nbformat as nbf

def convert_py_to_ipynb(py_file_path, ipynb_file_path):
    with open(py_file_path, 'r') as file:
        python_code = file.read()

    # Split the Python code into cells based on sections and subsections
    def split_into_cells(code):
        lines = code.split('\n')
        cells = []
        current_cell = []
        
        for line in lines:
            if line.startswith('# %%') or line.startswith('# %%%'):
                if current_cell:
                    cells.append('\n'.join(current_cell))
                    current_cell = []
            current_cell.append(line)
        
        if current_cell:
            cells.append('\n'.join(current_cell))
        
        return cells

    # Create a new Jupyter notebook object
    nb = nbf.v4.new_notebook()

    # Split the code into cells based on sections and subsections
    code_cells = split_into_cells(python_code)

    # Add each code cell to the notebook
    for cell_code in code_cells:
        nb.cells.append(nbf.v4.new_code_cell(cell_code))

    # Write the updated Jupyter notebook to a file
    with open(ipynb_file_path, 'w') as notebook_file:
        nbf.write(nb, notebook_file)

# File paths
py_file_path = 'PMF_DLP_Fit_Shap_Uni_Npp.py'  # Your Python file
ipynb_file_path = 'Updated_PMF_DLP_Fit_Shap_Uni_Npp1.ipynb'  # Desired Jupyter Notebook file

# Convert the .py file to .ipynb
convert_py_to_ipynb(py_file_path, ipynb_file_path)
print(f"Converted {py_file_path} to {ipynb_file_path}")
