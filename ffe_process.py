import tkinter as tk
from tkinter import filedialog
from tabulate import tabulate
import csv

import numpy as np
import math

from scipy.stats import t, f
from scipy.optimize import minimize


# Console interface to get ffe tables data
def ffe_tables_interface():
    # Convert string value to float
    def to_float(element):
        if element is None: 
            return False
        try:
            return float(element)
        except ValueError:
            return None

    # Read data from a csv-file
    def read_csv(file_path):
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = [[to_float(x) for x in row] for row in reader]
        return data

    # Prompt user to fill a matrix
    def fill_table(k, n, text = "Row"):
        table = []
        for i in range(k):
            row_str = input(f"{text} {i+1}: Enter {n} values (separated by commas): ")
            row = [to_float(x) for x in row_str.split(',')]
            table.append(row)
        return table

    # Fill missing values with None for size consistency
    def fill_missing_with_none(table):
        max_len = max([len(row) for row in table])
        for row in table:
            for i in range(max_len - len(row)):
                row.append(None)
        return table

    # Remove rows that have None values
    def remove_none_rows(table):
        to_remove = []
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == None:
                    if j not in to_remove:
                        to_remove.append(j)
        
        count = 0
        for index in to_remove:
            for i in range(len(table)):
                table[i].pop(index - count)
            count += 1

        return table, count

    # Remove rows for size consistency between tables
    def resize_for_consistency(x_table, y_table):
        xN = len(x_table[0])
        yN = len(y_table[0])

        x_to_remove_count = xN - yN
        y_to_remove_count = yN - xN

        if x_to_remove_count > 0:
            for i in range(len(x_table)):
                x_table[i] = x_table[i][0:yN]
        elif y_to_remove_count > 0:
            for i in range(len(y_table)):
                y_table[i] = y_table[i][0:xN]

        return x_table, y_table, x_to_remove_count, y_to_remove_count

    # Get x natural values table
    def get_x_natural(factors):
        K = len(factors)
        N = 2**K

        table = np.zeros((K, N))

        for i in range(N):
            for j in range(K):
                if math.ceil((i + 1) / 2**j) % 2 == 1:
                    table[j][i] = factors[j][0] - factors[j][1]
                else:
                    table[j][i] = factors[j][0] + factors[j][1]

        return table

    # Get x natural values table
    def get_x_coded(x_natural, factors):
        K = len(x_natural)
        N = len(x_natural[0])

        table = np.zeros((K, N))

        wrong_encoded = 0
        for i in range(N):
            for j in range(K):
                table[j][i] = round((x_natural[j][i] - factors[j][0]) / factors[j][1], 10)
                if (table[j][i] != -1 and table[j][i] != 1):
                    wrong_encoded += 1

        return table, wrong_encoded

    # Get factors values table
    def get_factors(x_natural):
        K = len(x_natural)
        
        factors = np.zeros((K, 2))

        for i in range(K):
            x_nat_i = list(filter(lambda x: x != None, x_natural[i]))
            main_lvl = (max(x_nat_i) + min(x_nat_i)) / 2
            chng_int = max(x_nat_i) - main_lvl
            factors[i] = [main_lvl, chng_int]

        return factors


    print("===== Building Linear Model for Full Factorial Experiment (FFE) =====\n")

    # Prompt user for csv-file
    answer = input("Do you have a csv-file with data for the FFE table? (yes/no) ")

    # Read data from csv-file
    if answer.lower()[0] == "y":
        # Select and read first csv-file
        print("\nChoose first csv-file with factors table ")
        root = tk.Tk()
        root.withdraw()
        file_path1 = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        root.destroy()
        print("csv-file with factors: ", file_path1)
        x_natural = read_csv(file_path1)
        
        factors = get_factors(x_natural)

        # Select and read second csv-file
        print("\nChoose second csv-file with experiment results table ")
        root = tk.Tk()
        root.withdraw()
        file_path2 = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        root.destroy()
        print("csv-file with experiments results: ", file_path1)
        y_results = read_csv(file_path2)
    
    # Read data from user console input
    else:
        # Prompt user about FFE table sizes
        K = int(input("\nEnter value of K (number of factors): "))
        if (K < 2):
            K = 2
            print("WARNING: your value was less than 2, so it was assigned 2 automatically\n")
    
        M = int(input("Enter value of M (number of parallel experiments): "))
        if (M < 1):
            M = 1
            print("WARNING: your value was less than 1, so it was assigned 1 automatically")
       
        N = 2**K
        print(f"\nThe FFE size is N = 2^K = {N}")
    
        # Prompt user to fill tables
        manual_fill = input("\nDo you want to fill values for factors table manually? (yes/no) ")
        if manual_fill.lower()[0] == "y":
            print("\nEnter factors experiment values: [x1], [x2], ..., [xN]")
            x_natural = fill_table(K, N, "Factor values")
            factors = get_factors(x_natural)
        else:
            print("\nEnter factors values: [main level], [change interval]")
            factors = fill_table(K, 2, "Factor")
            x_natural = get_x_natural(factors)
        
        print("\nEnter experiments values: [y1], [y2], ..., [yN]")
        y_results = fill_table(M, N, "Experiment results")


    # Transforming data for size consistency
    print()
    x_natural, rem_count = remove_none_rows(fill_missing_with_none(x_natural))
    y_results = fill_missing_with_none(y_results)
    if rem_count > 0:
        print(f"WARNING: x_natural has None values in {rem_count} rows that were removed")

    x_natural, y_results, x_rem_count, y_rem_count = resize_for_consistency(x_natural, y_results)
    if x_rem_count > 0:
        print(f"WARNING: x_natural has {x_rem_count} rows more than y_results, these last rows were removed")
    if y_rem_count > 0:
        print(f"WARNING: y_results has {y_rem_count} rows more than x_natural, these last rows were removed")

    # Get coded x values
    x_coded, wrong_encoded = get_x_coded(x_natural, factors)
    if wrong_encoded > 0:
        print(f"WARNING: {wrong_encoded} values couldn't be encoded correctly, linear model might not work correctly")

    # Get true sizes of table
    K = len(x_natural)
    N = len(x_natural[0])
    M = len(y_results)

    if M > 1:
        # Ask user about running significance and adequacy tests
        run_sign_test = input("Do you want to run significance test? (yes/no) ").lower()[0] == "y"
        run_adeq_test = input("Do you want to run adequacy test? (yes/no) ").lower()[0] == "y"
        
        if run_sign_test or run_adeq_test:
            if any(any(x is None for x in row) for row in y_results):
                print(f"WARNING: some of y_results values are None, tests results might not be correct")
    else:
        run_sign_test = False
        run_adeq_test = False

    # In case of not full data for ffe
    if N < 2**K:
        print(f"WARNING: for entered data N ({N}) is less than 2^K (2^{K}), linear model might not work correctly")
    if any(any(x is None for x in row) for row in x_natural):
        print(f"WARNING: some of x_natural values are None, linear model might not work correctly")

    print("\n=====================================================================\n")

    return x_natural, x_coded, y_results, factors, run_sign_test, run_adeq_test


# Run FFE
def ffe_solver(x_natur, x_coded, y, factors, run_sign_test, run_adeq_test):
    # Get mean for y results
    def get_mean_y(y):
        y_means = [None] * len(y[0])
    
        for i in range(len(y[0])):
            y_sum = 0
            N = len(y)
            for k in range(len(y)):
                if y[k][i] != None:
                    y_sum += y[k][i]
                else:
                    N -= 1
            y_means[i] = y_sum / N
    
        return y_means

    # Get squered differences for y results
    def get_sq_diffs(y, y_means):
        y_sq_diffs = [None] * len(y[0])
    
        for i in range(len(y[0])):
            y_sq_sum = 0
            for k in range(len(y)):
                if y[k][i] != None:
                    y_sq_sum += (y_means[i] - y[k][i]) ** 2
            y_sq_diffs[i] = y_sq_sum / (len(y) - 1)
    
        return y_sq_diffs

    # Get FFE table 
    def ffe_table(factors = [], x_coded = [], y = [], ym = [], s = []):
        leng = 0
        if len(factors) > 0:
            leng += len(factors)
        if len(x_coded) > 0:
            leng += len(x_coded)
        if len(y) > 0:
            leng += len(y)
        if len(ym) > 0:
            leng += 1
        if len(s):
            leng += 1
        table = np.zeros((len(ym), leng))
        headers = [None] * leng

        i = 0
        if len(x_coded):
            for j in range(len(x_coded)):
                table[:, i] = x_coded[j]
                headers[i] = f'x{j+1} (c)'
                i += 1
        if len(factors):
            for j in range(len(factors)):
                table[:, i] = factors[j]
                headers[i] = f'x{j+1}'
                i += 1
        if len(y):
            for j in range(len(y)):
                table[:, i] = y[j]
                headers[i] = f'y{j+1}'
                i += 1
        if len(ym):
            table[:, i] = ym
            headers[i] = 'y_'
            i += 1
        if len(s):
            table[:, i] = s
            headers[i] = 's^2'
    
        return table, headers

    # Get b polinomial coefficients
    def get_b(x, y):
        b = [None] * (int)(math.ceil(math.log2(len(y)) + 1))
        headers = []
    
        for i in range(len(b)):
            if i == 0:
                b[i] = np.mean(y)
            else:
                b[i] = (np.array(x[i-1]) @ np.array(y)) / len(y)
            
            headers.append(f'b{i}')
        
        return b, headers

    # Get string of polinomial equation
    def get_pol_mod_pr(b):
        pol_mod_pr = "y = "
        for i in range(len(b)):
            if i == 0:
                pol_mod_pr += f"{b[i]:.4f}"
            else:
                sign = "" if b[i] < 0 else " +"
                pol_mod_pr += f"{sign} {b[i]:.4f}x{i}"
        return pol_mod_pr

    # Run significance test for b coefficients
    def check_significance(s, b):
        df = len(s) * (math.log2(len(s)) - 1)
        conf_lvl = 0.95
        t_val = t.ppf(q=(1 + conf_lvl) / 2, df=df)
        sb = math.sqrt(sum(s) / math.log2(len(s))) / len(s)
    
        print(tabulate([[df, conf_lvl, t_val, sb]], headers=["F", "p", "t", "Sb"], tablefmt="simple_grid"))

        tsb = t_val * sb

        sign_checks = []
        sign_checks_table = []
        for i in range(len(b)):
            significant = tsb < math.fabs(b[i])
            sign_checks.append(significant);
            sign_checks_table.append([f't*Sb={tsb:.4f} {"<" if significant else ">"} b{i}={math.fabs(b[i]):.4f}', f'{"" if significant else "not "}significant'])
        
        print(tabulate(sign_checks_table, tablefmt="simple_grid"))
    
        return significant

    # Run adequacy (Fisher) test for b coefficients
    def check_adequacy(x, ym, s, b):
        yp = []
        yp_headers = []
        for i in range(len(s)):
            ypi = 0
            for j in range(len(b)):
                if j == 0:
                    ypi += b[j]
                else:
                    ypi += b[j] * x[j-1][i]
            yp.append(ypi)
            yp_headers.append(f'y^p {i+1}')
    
        print(tabulate([yp], headers=yp_headers, tablefmt="simple_grid"))

        s_add = 0
        s_y = 0
        for i in range(len(yp)):
            s_add += (ym[i] - yp[i])**2
            s_y += s[i]

        conf_lvl = 0.95
        dfn = 2**math.ceil(math.log2(len(s))) - len(b) # math.ceil(math.log2(len(s))) == len(s) if all experiments filled
        dfd = 2**math.ceil(math.log2(len(s))) * (math.ceil(math.log2(len(s))) - 1)
        Ft = f.ppf(q=conf_lvl, dfn=dfn, dfd=dfd)

        s_add /= dfn
        s_y /= dfd

        print(tabulate([[s_add, s_y]], headers=["S^2 ad", "S^2 y"], tablefmt="simple_grid"))

        F = s_y / s_add

        adequate = F < Ft

        print(tabulate([[dfn, dfd, conf_lvl, Ft, F]], headers=["dfn", "dfd", "p", "Ft", "F"], tablefmt="simple_grid"))
        print(tabulate([[f'{F:.4f} {"<" if adequate else ">"} {Ft:.4f}', f'{"" if adequate else "not "}adequate']], tablefmt="simple_grid"))

        return adequate

    # Get b polinomial coefficients in natural scale
    def get_nat_b(b, nat_table):
        nat_b = [None] * len(b)
        nat_headers = []

        for i in range(len(b)):
            if i == 0:
                nat_b[i] = b[i]
            else:
                nat_b[i] = b[i] / nat_table[i-1][1]
                nat_b[0] -= b[i] * (nat_table[i-1][0] / nat_table[i-1][1])
    
            nat_headers.append(f'nat_b{i}')

        return nat_b, nat_headers

    # Run optimization for found polinomial b coefficients
    def optimize_func(factors_ch, b):
        def f(x):
            res = b[0]
            for i in range(len(x)):
                res += x[i] * b[i+1]
            return res

        def get_bounds(factors_ch):
            bounds = []
            for i in range(len(factors_ch)):
                bounds.append((factors_ch[i][0] - factors_ch[i][1], factors_ch[i][0] + factors_ch[i][1]))
            return bounds
    
        x0 = np.array(factors_ch)[:,0]
        bounds = get_bounds(factors_ch)

        result = minimize(f, x0, bounds=bounds)

        table = []
        table.append(["Function", get_pol_mod_pr(b)])
        for i in range(len(bounds)):
            table.append([f'x{i+1} bounds', f'({bounds[i][0]}; {bounds[i][1]})'])
        table.append(["y optimized", f'{result.fun}'])
        for i in range(len(result.x)):
            table.append([f'x{i+1}', f'{result.x[i]}'])

        return table


    ym = None
    s = None
    
    if len(y) > 1:
        ym = get_mean_y(y)
        s = get_sq_diffs(y, ym)
        table, headers = ffe_table(x_natur, x_coded, y, ym, s)
    else:
        ym = y[0]
        table, headers = ffe_table(x_natur, x_coded, ym=ym)

    print("FULL FACTOR EXPERIMENT TABLE")
    print(tabulate(table, headers=headers, tablefmt="simple_grid"))

    b, b_headers = get_b(x_coded, ym)
    print("\nCOEFFICIENTS OF POLINOMIAL MODEL")
    print(tabulate([b], headers=b_headers, tablefmt="simple_grid"))
    print(tabulate([[get_pol_mod_pr(b)]], headers=["Polinomial Model of Process"], tablefmt="simple_grid"))

    if s != None:
        if (run_sign_test):
            print("\nSIGNIFICANCE CHECK FOR COEFFICIENTS")
            check_significance(s, b)
        
        if (run_adeq_test):
            print("\nADEQUACY CHECK")
            check_adequacy(x_coded, ym, s, b)

    nat_b, nat_b_headers = get_nat_b(b, factors)
    print("\nNATURAL VALUES OF POLINOMIAL MODEL")
    print(tabulate([nat_b], headers=nat_b_headers, tablefmt="simple_grid"))
    print(tabulate([[get_pol_mod_pr(nat_b)]], headers=["Polinomial Model of Process"], tablefmt="simple_grid"))

    opt_results_table = optimize_func(factors, nat_b)
    print("\nOPTIMIZATION OF FUNCTION")
    print(tabulate(opt_results_table, tablefmt="simple_grid"))


if __name__ == "__main__":
    run_ffe = True
    while run_ffe:
        x_natur, x_coded, y, factors, run_sign_test, run_adeq_test = ffe_tables_interface()
        ffe_solver(x_natur, x_coded, y, factors, run_sign_test, run_adeq_test)
        
        print("\n---------------------------------------------------------------------\n")
        run_ffe = input("Do you want to run one more FFE? (yes/no) ").lower()[0] == "y"
        print("\n---------------------------------------------------------------------\n")
    
    print("Okay, Bye")
