import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt


def box_cox(x, lambda_range = 5):

    lambdas = np.arange(-5,6)

    y = np.zeros((len(x), len(lambdas)))
    for i, l in enumerate(lambdas):
        if l == 0:
            y[:, i] = np.log(np.maximum(x, np.exp(-5)))
        else:
            y[:, i] = (np.power(x, l) - 1) / l

    k = np.zeros(len(lambdas))
    p_values = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        k[i], p_values[i] = stats.normaltest(y[:, i])
        if np.isnan(k[i]):
            k[i] = np.exp(10)
            p_values[i] = np.exp(-10)

    p_value = p_values[np.argmin(k)]
    lambda_opt = lambdas[np.argmin(k)]

    if lambda_opt == 0:
        transform = lambda x: np.log(np.maximum(x, np.exp(-5)))
    elif lambda_opt == 1:
        transform = lambda x: x
    else:
        transform = lambda x: (np.power(x, lambda_opt) - 1) / lambda_opt

    y_transf = transform(x)

    mean = np.mean(y_transf)
    std = np.std(y_transf)

    return mean, std, p_value, lambda_opt, transform

def one_hot_enconding(input, output, convert_columns, filt_columns = None):
    df = pd.read_csv(input)
    if filt_columns is not None:
        df = df[filt_columns]

    for c, column in enumerate(convert_columns):
        if column in df.columns:
            id = df[column]
            colunas_dummy = pd.get_dummies(id, prefix='%d-Classe'%(c))
            df = pd.concat([colunas_dummy, df.drop(column, axis=1)], axis=1)

    if output is not None:
        df.to_csv(output, index=False)
    return df.columns

class Selector():

    def __init__(self, filename, input_labels, output_labels) -> None:
        self.filename = filename
        self.df = pd.read_csv(filename)
        self.inputs = self.df[input_labels]
        self.outputs = self.df[output_labels]
        self.df = pd.concat([self.inputs, self.outputs], axis=1)

        self.input_labels = input_labels
        self.output_labels = output_labels

        self.n_samples = self.inputs.shape[0]
        self.n_inputs = self.inputs.shape[1]
        self.n_outputs = self.outputs.shape[1]

        self.corr = np.corrcoef(self.df.values.T)

    def export_output_corr(self, filename, min_score=None, margin = 0.02):
        if min_score == None:
            min_score = 2/np.sqrt(self.n_samples)

        selected_inputs = []

        output_table = [[""] * (self.n_outputs+1) for _ in range(self.n_inputs+1)]
        for i, input in enumerate(self.input_labels):
            for j in range(self.n_outputs):
                if abs(self.corr[i,self.n_inputs + j]) > min_score:
                    output_table[i+1][j+1] = "\\textbf{" + '{:.2f}'.format(self.corr[i,self.n_inputs + j]) + '}'

                    if not input in selected_inputs:
                        selected_inputs.append(input)

                elif abs(self.corr[i,self.n_inputs + j]) > min_score - margin:
                    output_table[i+1][j+1] = "\\textit{" + '{:.2f}'.format(self.corr[i,self.n_inputs + j]) + '}'

                    if not input in selected_inputs:
                        selected_inputs.append(input)

                else:
                    output_table[i+1][j+1] = '{:.2f}'.format(self.corr[i,self.n_inputs + j])

        for i in range(self.n_inputs):
            output_table[i+1][0] = self.inputs.columns[i]
        for j in range(self.n_outputs):
            output_table[0][j+1] = self.outputs.columns[j]

        for row in output_table:
            for i in range(len(row)):
                row[i] = row[i].replace(".", ",")

        with open(filename, 'w') as f:
            f.write(tabulate(output_table, headers='firstrow', floatfmt=".2f", tablefmt='latex_raw'))

        return selected_inputs

    def export_input_corr(self, filename, min_score=None, fake_headers=False, margin = 0.02):
        if min_score == None:
            min_score = 2/np.sqrt(self.n_samples)

        input_table = [[""] * (self.n_inputs + 1) for _ in range(self.n_inputs + 1)]
        for i in range(self.n_inputs):
            for j in range(i+1):
                if abs(self.corr[i,j]) > min_score:
                    input_table[i+1][j+1] = "\\textbf{" + '{:.2f}'.format(self.corr[i,j]) + '}'
                elif abs(self.corr[i,j]) > min_score - margin:
                    input_table[i+1][j+1] = "\\textit{" + '{:.2f}'.format(self.corr[i,j]) + '}'
                else:
                    input_table[i+1][j+1] = '{:.2f}'.format(self.corr[i,j])

        for i in range(self.n_inputs):
            if fake_headers:
                input_table[i+1][0] = "$x_{%d}$"%(i)
                input_table[0][i+1] = "$x_{%d}$"%(i)
            else:
                input_table[i+1][0] = self.inputs.columns[i]
                input_table[0][i+1] = self.inputs.columns[i]

        for row in input_table:
            for i in range(len(row)):
                row[i] = row[i].replace(".", ",")

        with open(filename, 'w') as f:
            f.write(tabulate(input_table, headers='firstrow', floatfmt=".2f", tablefmt='latex_raw'))

    def transform_inputs(self, output_path, ignore_column = [], prefix = ""):

        transform_results = {}

        if prefix != "":
            prefix = prefix + "_"

        hist_path = os.path.join(output_path,"histograms")
        os.makedirs(hist_path, exist_ok=True)

        for col in self.inputs.columns:

            if self.n_samples >= 100:
                n_bins = np.round(5 * np.log10(self.n_samples))
            else:
                n_bins = np.round(np.sqrt(self.n_samples))

            plt.hist(self.inputs[col], bins=int(n_bins))
            plt.xlabel(col)
            plt.ylabel('Quantidade')
            plt.savefig(os.path.join(hist_path, prefix + col + ".png"))
            plt.close()

            if col in ignore_column:
                continue

            try:

                mean, std, p_value, lambda_opt, transform = box_cox(self.inputs[col], col)

                self.inputs[col] = (transform(self.inputs[col]) - mean)/std

                if lambda_opt != 1:
                    plt.hist(self.inputs[col], bins=int(n_bins))
                    plt.xlabel(col + "(" + r'$\lambda$' + " = {:d})".format(lambda_opt))
                    plt.ylabel('Quantidade')
                    plt.savefig(os.path.join(hist_path, prefix + col + "_transformed.png"))
                    plt.close()

                transform_results[col] = str(lambda_opt)

            except ValueError:
                min = np.min(self.inputs[col])
                max = np.max(self.inputs[col])
                transform = lambda x: (x-min-((max-min)/2))/((max-min)/2)
                self.inputs[col] = transform(self.inputs[col])

                plt.hist(self.inputs[col], bins=int(n_bins))
                plt.xlabel(col + "(normalizado)")
                plt.ylabel('Quantidade')
                plt.savefig(os.path.join(hist_path, prefix + col + "_transformed.png"))
                plt.close()

                transform_results[col] = {'min': str(min), 'max': str(max)}

            except:
                print(col, " not compatible")

        path, rel_filename = os.path.split(self.filename)
        filename, extension = os.path.splitext(rel_filename)

        out_df = pd.concat([self.inputs, self.outputs], axis=1)
        out_df.to_csv(os.path.join(output_path, filename + "_norm.csv"), index=False)
        print("gerado arquivo: ", os.path.join(output_path, filename + "_norm.csv"))

        with open(os.path.join(output_path, filename + "_transformed.json"), 'w') as f:
            json.dump(transform_results, f, indent=4)
        return transform_results

    def save(self, filename, input_labels, output_labels):
        df = pd.concat([self.inputs[input_labels], self.outputs[output_labels]], axis=1)
        df.to_csv(filename, index=False)


