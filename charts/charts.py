import cv2
from read_chart.read_chart import ChartReader
from charts_axes_detect.chart_run import process_single_file,CONFIG
from data_analyzer.symbolic_regressor import SymbolicRegressor
import numpy as np
import os
class ChartAnalyzer:
    def __init__(self, chart_path):
        self.chart_path = chart_path
        self.chart = cv2.imread(chart_path)
        self.data_analyzer = SymbolicRegressor()
        
    def compare_with_csv(self, csv_path):
        return self.data_analyzer.compare_with_csv(csv_path)

    def analyze(self):
        # Read the chart image  
        axes,axes_data = process_single_file(self.chart_path,config=CONFIG,output_dir=CONFIG['output_directory'])
        # Extract bounding boxes of axes
        bbox_x = axes_data["horizontal_axes"][0]["bbox"]
        bboy_y = axes_data["vertical_axes"][0]["bbox"]
        
        
        chart_reader = ChartReader(self.chart, axis_x=axes['horizontal_axes'], axis_y=axes['vertical_axes'], bbox_x=bbox_x, bbox_y=bboy_y)    
        data_points = chart_reader.extract_data_points()
        x_values, y_values = zip(*data_points)
        # Convert to numpy arrays
        x_values = np.array(x_values).reshape(-1, 1)
        y_values = np.array(y_values)
        self.data_analyzer.set_data(x_values,y_values)
        self.data_analyzer.fit()

        formula, latex_formula = self.data_analyzer.get_formula()
        print("Najlepszy wzór:", formula)
        x_base= axes["horizontal_axes"][0]["logarithm_base"]
        y_base= axes["vertical_axes"][0]["logarithm_base"]
        self.data_analyzer.plot(x_base,y_base)    # Perform symbolic regression on the extracted data points
        metrics = self.data_analyzer.score()
        print("Metryki dopasowania:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        export_path = "results/latex/overleaf_export.tex"
        self.export_to_overleaf(export_path,(axes["horizontal_axes"][0]["values"][0], axes["horizontal_axes"][0]["values"][-1]), len(x_values))
    def export_to_overleaf(self, output_path, domain=(-10, 10), samples=100):
        """
        Exports the symbolic regression results to an Overleaf-compatible LaTeX file.
        
        Args:
            output_path: Path to save the LaTeX file
            domain: Tuple of (min, max) x values for plotting
            samples: Number of points to sample within the domain
        """
        formula, latex_formula = self.data_analyzer.get_formula()
        print(formula)
        domain_min, domain_max = domain
        #Check if path exists, if not create it
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #replace ** with ^ in the formula for pf compatibility
        formula = str(formula).replace("**", "^")
        formula = str(formula).replace("x0", "x")
        with open(output_path, 'w') as f:
            f.write(f"""
                    \\documentclass{{article}}
                    \\usepackage[margin=0.25in]{{geometry}}
                    \\usepackage{{pgfplots}}
                    \\pgfplotsset{{width=10cm,compat=1.9}}

                    \\begin{{document}}
                    \\begin{{tikzpicture}}
                    \\begin{{axis}}[
                        axis lines = left,
                        xlabel = \\(x\\),
                        ylabel = \\(f(x)\\),
                    ]
                    % Fitted formula from symbolic regression
                    \\addplot [
                        domain={domain_min}:{domain_max}, 
                        samples={samples}, 
                        color=red,
                    ]
                    {{{formula}}};
                    \\addlegendentry{{\\({latex_formula}\\)}}

                    % Original data points
                    \\addplot[only marks, mark=o, mark size=1.5pt, color=blue] 
                        table {{
                        % Here you could add your actual data points if needed
                    }};
                    \\addlegendentry{{Data points}}

                    \\end{{axis}}

                    \\end{{tikzpicture}}
                    \\end{{document}}
                    """)
        print(f"LaTeX file exported to {output_path}")

def main():
    chart_path = "charts_examples/1.JPG"  # Replace with your chart image path
    csv_path = "dane/1.csv"  # Replace with your CSV file path
    analyzer = ChartAnalyzer(chart_path)
    result = analyzer.analyze()
    comparison_metrics = analyzer.compare_with_csv(csv_path)

if __name__ == "__main__":
    main()