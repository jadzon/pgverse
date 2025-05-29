import cv2
from read_chart.read_chart import ChartReader
from charts_axes_detect.chart_run import process_single_file,CONFIG
from data_analyzer.symbolic_regressor import SymbolicRegressor
import numpy as np
class ChartAnalyzer:
    def __init__(self, chart_path):
        self.chart_path = chart_path
        self.chart = cv2.imread(chart_path)
        self.data_analyzer = SymbolicRegressor()

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
        self.data_analyzer.plot()    # Perform symbolic regression on the extracted data points
        metrics = self.data_analyzer.score()
        print("Metryki dopasowania:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


def main():
    chart_path = "charts_examples/test5.JPG"  # Replace with your chart image path
    analyzer = ChartAnalyzer(chart_path)
    result = analyzer.analyze()


if __name__ == "__main__":
    main()