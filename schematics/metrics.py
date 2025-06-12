import json
import os
import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class SchematicMetricsAnalyzer:
    def __init__(self, raw_data_path, user_data_path=None):
        """Initialize analyzer with paths to data files"""
        self.raw_data_path = raw_data_path
        self.user_data_path = user_data_path
        self.raw_data = self._load_json(raw_data_path)
        self.user_data = self._load_json(user_data_path) if user_data_path else None
        
    def _load_json(self, file_path):
        """Load JSON data from file"""
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
        
    def analyze(self):
        """Run complete analysis and return all metrics"""
        results = {}
        
        # Basic component metrics
        results["component_count"] = len(self.raw_data)
        results["component_types"] = self._component_types()
        
        # Connection metrics
        results["connection_metrics"] = self._analyze_connections()
        results["isolated_components"] = self._find_isolated_components()
        results["connection_density"] = self._connection_density()
        
        # Layout metrics
        results["spatial_metrics"] = self._analyze_spatial_layout()
        
        # Text and label metrics
        results["text_metrics"] = self._analyze_text_labels()
        
        # Consistency checks
        results["consistency_checks"] = self._check_consistency()
        
        return results
    
    def _component_types(self):
        """Count components by type"""
        types = Counter()
        for comp_id, comp_data in self.raw_data.items():
            block_type = comp_data.get("block", "Unknown")
            types[block_type] += 1
        return dict(types)
    
    def _analyze_connections(self):
        """Analyze connections between components"""
        metrics = {}
        
        # Connection counts
        total_connections = 0
        connection_counts = []
        
        for comp_id, comp_data in self.raw_data.items():
            conn_count = len(comp_data.get("connections", []))
            connection_counts.append(conn_count)
            total_connections += conn_count
        
        # Divide by 2 because each connection is counted twice (A→B and B→A)
        metrics["total_connections"] = total_connections // 2
        metrics["avg_connections_per_component"] = np.mean(connection_counts) if connection_counts else 0
        metrics["max_connections"] = max(connection_counts) if connection_counts else 0
        metrics["min_connections"] = min(connection_counts) if connection_counts else 0
        
        # Check for bidirectional consistency errors
        metrics["bidirectional_errors"] = self._check_bidirectional_connections()
        
        return metrics
    
    def _check_bidirectional_connections(self):
        """Check if all connections are properly bidirectional"""
        errors = []
        
        for comp_id, comp_data in self.raw_data.items():
            for connected_id in comp_data.get("connections", []):
                connected_id = str(connected_id)
                if connected_id in self.raw_data:
                    if int(comp_id) not in self.raw_data[connected_id].get("connections", []):
                        errors.append(f"Component {comp_id} connects to {connected_id}, but not vice versa")
                else:
                    errors.append(f"Component {comp_id} connects to non-existent component {connected_id}")
        
        return errors
    
    def _find_isolated_components(self):
        """Find components with no connections"""
        isolated = []
        for comp_id, comp_data in self.raw_data.items():
            if not comp_data.get("connections", []):
                isolated.append(comp_id)
        return isolated
    
    def _connection_density(self):
        """Calculate connection density (actual/potential connections)"""
        n = len(self.raw_data)
        if n <= 1:
            return 0
        
        potential_connections = n * (n - 1) / 2
        actual_connections = sum(len(comp_data.get("connections", [])) for _, comp_data in self.raw_data.items()) / 2
        
        return actual_connections / potential_connections if potential_connections > 0 else 0
    
    def _analyze_spatial_layout(self):
        """Analyze spatial layout and positioning of components"""
        metrics = {}
        positions = []
        
        for comp_id, comp_data in self.raw_data.items():
            coords = comp_data.get("coordinates", {})
            if coords:
                center_x = (coords.get("x1", 0) + coords.get("x2", 0)) / 2
                center_y = (coords.get("y1", 0) + coords.get("y2", 0)) / 2
                positions.append((center_x, center_y))
        
        if positions:
            # Calculate bounding box
            min_x = min(pos[0] for pos in positions)
            max_x = max(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            max_y = max(pos[1] for pos in positions)
            
            metrics["bounding_box"] = {
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y,
                "width": max_x - min_x,
                "height": max_y - min_y
            }
            
            # Calculate component density
            area = (max_x - min_x) * (max_y - min_y)
            metrics["component_density"] = len(positions) / area if area > 0 else 0
            
            # Calculate distances between components
            if len(positions) > 1:
                distances = []
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        dist = math.sqrt((positions[i][0] - positions[j][0])**2 + 
                                        (positions[i][1] - positions[j][1])**2)
                        distances.append(dist)
                
                metrics["avg_component_distance"] = np.mean(distances)
                metrics["min_component_distance"] = min(distances)
                metrics["max_component_distance"] = max(distances)
        
        return metrics
    
    def _analyze_text_labels(self):
        """Analyze text labels on components"""
        metrics = {}
        components_with_text = 0
        label_lengths = []
        
        for comp_id, comp_data in self.raw_data.items():
            texts = comp_data.get("texts", "")
            if texts and texts != "":
                components_with_text += 1
                
                # Handle different text formats
                if isinstance(texts, list):
                    for text_item in texts:
                        if isinstance(text_item, list) and len(text_item) > 0:
                            label_lengths.append(len(str(text_item[0])))
                elif isinstance(texts, str):
                    label_lengths.append(len(texts))
        
        metrics["labeled_component_percentage"] = (components_with_text / len(self.raw_data)) * 100 if self.raw_data else 0
        metrics["avg_label_length"] = np.mean(label_lengths) if label_lengths else 0
        
        return metrics
    
    def _check_consistency(self):
        """Check for consistency issues in the schematic"""
        issues = []
        
        # Check for duplicate component labels
        component_labels = defaultdict(list)
        for comp_id, comp_data in self.raw_data.items():
            comp_type = comp_data.get("block", "Unknown")
            texts = comp_data.get("texts", "")
            
            label = None
            if isinstance(texts, list):
                for text_item in texts:
                    if isinstance(text_item, list) and len(text_item) > 0:
                        label = str(text_item[0])
                        break
            elif isinstance(texts, str) and texts:
                label = texts
            
            if label:
                component_labels[(comp_type, label)].append(comp_id)
        
        # Find duplicates
        for (comp_type, label), comp_ids in component_labels.items():
            if len(comp_ids) > 1:
                issues.append(f"Duplicate label '{label}' for {comp_type}: {comp_ids}")
        
        # Check for missing connections
        for comp_id, comp_data in self.raw_data.items():
            connection_points = comp_data.get("connection_points", [])
            connections = comp_data.get("connections", [])
            
            if connection_points and not connections:
                issues.append(f"Component {comp_id} has connection points but no connections")
        
        return issues
    
    def compare_with_user_data(self):
        """Compare detected schematic with user-provided data"""
        if not self.user_data:
            return {"error": "No user data provided for comparison"}
        
        results = {}
        
        # Compare component counts and types
        results["component_count_match"] = len(self.raw_data) == len(self.user_data)
        
        raw_types = self._component_types()
        user_types = Counter()
        for comp_id, comp_data in self.user_data.items():
            block_type = comp_data.get("block", "Unknown")
            user_types[block_type] += 1
        
        results["component_types_match"] = raw_types == dict(user_types)
        results["type_differences"] = {
            k: {"raw": raw_types.get(k, 0), "user": user_types.get(k, 0)}
            for k in set(raw_types) | set(user_types)
            if raw_types.get(k, 0) != user_types.get(k, 0)
        }
        
        # Compare connections
        raw_connections = set()
        for comp_id, comp_data in self.raw_data.items():
            for conn in comp_data.get("connections", []):
                raw_connections.add((min(int(comp_id), conn), max(int(comp_id), conn)))
        
        user_connections = set()
        for comp_id, comp_data in self.user_data.items():
            for conn in comp_data.get("connections", []):
                user_connections.add((min(int(comp_id), int(conn)), max(int(comp_id), int(conn))))
        
        results["connections_match"] = raw_connections == user_connections
        results["missing_connections"] = list(raw_connections - user_connections)
        results["extra_connections"] = list(user_connections - raw_connections)
        
        # Count correctly identified objects
        correctly_identified = 0
        correctly_identified_by_type = defaultdict(int)
        
        # First, map components by position for matching
        raw_components_by_pos = {}
        for comp_id, comp_data in self.raw_data.items():
            coords = comp_data.get("coordinates", {})
            if coords:
                center_x = (coords.get("x1", 0) + coords.get("x2", 0)) / 2
                center_y = (coords.get("y1", 0) + coords.get("y2", 0)) / 2
                raw_components_by_pos[(center_x, center_y)] = (comp_id, comp_data)
        
        user_components_by_pos = {}
        for comp_id, comp_data in self.user_data.items():
            coords = comp_data.get("coordinates", {})
            if coords:
                center_x = (coords.get("x1", 0) + coords.get("x2", 0)) / 2
                center_y = (coords.get("y1", 0) + coords.get("y2", 0)) / 2
                user_components_by_pos[(center_x, center_y)] = (comp_id, comp_data)
        
        # Match components and check for correct identification
        for pos, (raw_id, raw_comp) in raw_components_by_pos.items():
            # Find closest user component
            closest_pos = min(user_components_by_pos.keys(), 
                             key=lambda p: ((p[0]-pos[0])**2 + (p[1]-pos[1])**2),
                             default=None)
            
            if closest_pos is None:
                continue
                
            # Calculate distance to closest component
            dist = math.sqrt((closest_pos[0]-pos[0])**2 + (closest_pos[1]-pos[1])**2)
            
            # Skip if too far (adjust threshold as needed)
            threshold = 20  # Arbitrary distance threshold
            if dist > threshold:
                continue
                
            user_id, user_comp = user_components_by_pos[closest_pos]
            
            # Check if component type matches
            if raw_comp.get("block") == user_comp.get("block"):
                # Component is correctly identified
                correctly_identified += 1
                correctly_identified_by_type[raw_comp.get("block", "Unknown")] += 1
        
        results["correctly_identified_count"] = correctly_identified
        results["correctly_identified_by_type"] = dict(correctly_identified_by_type)
        
        # Calculate identification accuracy
        total_user_components = len(self.user_data)
        results["identification_accuracy"] = (correctly_identified / total_user_components * 100) if total_user_components > 0 else 0
        
        return results
    
    def visualize_schematic(self, output_path=None):
        """Generate a visual representation of the schematic"""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Component centers and colors
        component_centers = {}
        component_colors = {
            "Resistor": "green",
            "Capacitor": "blue",
            "Voltage_Source": "red",
            "AC_Source": "orange",
            "BJT": "purple",
            "default": "gray"
        }
        
        # Draw components
        for comp_id, comp_data in self.raw_data.items():
            coords = comp_data.get("coordinates", {})
            comp_type = comp_data.get("block", "Unknown")
            
            if coords:
                x1, y1 = coords.get("x1", 0), coords.get("y1", 0)
                x2, y2 = coords.get("x2", 0), coords.get("y2", 0)
                width, height = x2 - x1, y2 - y1
                
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                component_centers[comp_id] = (center_x, center_y)
                
                color = component_colors.get(comp_type, component_colors["default"])
                rect = patches.Rectangle((x1, y1), width, height, linewidth=1, 
                                       edgecolor=color, facecolor='none', alpha=0.7)
                ax.add_patch(rect)
                
                # Add component label
                ax.text(center_x, center_y, f"{comp_id}:{comp_type}", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=8)
        
        # Draw connections
        drawn_connections = set()
        for comp_id, comp_data in self.raw_data.items():
            if comp_id in component_centers:
                start_x, start_y = component_centers[comp_id]
                
                for connected_id in comp_data.get("connections", []):
                    connected_id = str(connected_id)
                    
                    # Only draw each connection once
                    connection_key = tuple(sorted([comp_id, connected_id]))
                    if connection_key in drawn_connections:
                        continue
                    
                    drawn_connections.add(connection_key)
                    
                    if connected_id in component_centers:
                        end_x, end_y = component_centers[connected_id]
                        ax.plot([start_x, end_x], [start_y, end_y], 'k-', linewidth=0.5, alpha=0.6)
        
        # Set limits and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Schematic Visualization')
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

def main():
    # Define paths
    
    user_data_path = None
    raw_data_path = "main_results/circuitikz/circuit_raw.json"
    metrics_json_dir = "metrics_jsons"
    if os.path.exists(metrics_json_dir):
        user_files = [os.path.join(metrics_json_dir, f) for f in os.listdir(metrics_json_dir) 
                    if f.endswith('.json')]
        if user_files:
            user_data_path = user_files[0]
    
    print(f"Found user data: {user_data_path}" if user_data_path else "No user data found.")
    # Output directory
    output_dir = "metrics_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create analyzer and run analysis
    analyzer = SchematicMetricsAnalyzer(raw_data_path, user_data_path)
    results = analyzer.analyze()
    
    # Generate report
    report_path = os.path.join(output_dir, "schematic_metrics_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization
    vis_path = os.path.join(output_dir, "schematic_visualization.png")
    analyzer.visualize_schematic(vis_path)
    
    # Compare with user data if available
    if analyzer.user_data:
        comparison = analyzer.compare_with_user_data()
        comp_path = os.path.join(output_dir, "user_comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return results

if __name__ == "__main__":
    main()