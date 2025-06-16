import json
import os
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import hashlib

# ==============================================================================
#  STEP 1: DEFINE THE VISUAL DATA FLOW REPORTER CLASS
#  (This is the class developed in the previous step)
# ==============================================================================

class VisualDataFlowReporter:
    def __init__(self):
        self.search_results = []
        self.filtered_results = []
        self.answers = []
        self.result_connections = {}
        # Store row-level connection data to draw arrows
        self.flow_map = {
            'search_to_filtered': [], # List of (source_row, dest_row) tuples
            'filtered_to_answer': []  # List of (source_row, dest_row) tuples
        }

    def create_content_fingerprint(self, result_dict):
        key_fields = []
        for field in ['title', 'url', 'content', 'description', 'text']:
            if field in result_dict and result_dict[field]:
                key_fields.append(str(result_dict[field]).strip().lower())
        
        combined_content = '|||'.join(key_fields)
        return hashlib.md5(combined_content.encode('utf-8')).hexdigest()[:8]
    
    def load_and_analyze_data(self, json_file_paths):
        print("Loading data and building relationship map...")
        for file_path in json_file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._process_search_results(data, file_path)
            self._process_filtered_results(data, file_path)
            self._process_answers(data, file_path)
        
        print(f"Processed {len(self.search_results)} search results, {len(self.filtered_results)} filtered, and {len(self.answers)} answers.")

    def _process_search_results(self, data, file_path):
        search_results = data.get('search_results', [])
        for idx, result in enumerate(search_results):
            fingerprint = self.create_content_fingerprint(result)
            enhanced_result = {
                'fingerprint': fingerprint, 'source_file': Path(file_path).name,
                'position': idx + 1, 'row': 3 + len(self.search_results),
                **result
            }
            self.search_results.append(enhanced_result)
    
    def _process_filtered_results(self, data, file_path):
        filtered_results = data.get('filtered_search_results', [])
        for result in filtered_results:
            fingerprint = self.create_content_fingerprint(result)
            original_result = self._find_original_by_fingerprint(fingerprint)
            if original_result:
                dest_row = 3 + len(self.filtered_results)
                enhanced_result = {
                    'fingerprint': fingerprint, 'source_file': Path(file_path).name,
                    'original_position': original_result['position'], 'row': dest_row,
                    **result
                }
                self.filtered_results.append(enhanced_result)
                self.flow_map['search_to_filtered'].append((original_result['row'], dest_row))
                if fingerprint not in self.result_connections: self.result_connections[fingerprint] = {}
                self.result_connections[fingerprint]['filtered'] = True

    def _process_answers(self, data, file_path):
        answers = data.get('answers_to_questions', [])
        for answer in answers:
            dest_row = 3 + len(self.answers)
            referenced_fingerprints = []
            for ref_result in answer.get('referenced_results', []):
                fingerprint = self.create_content_fingerprint(ref_result)
                referenced_fingerprints.append(fingerprint)
                source_result = self._find_filtered_by_fingerprint(fingerprint)
                if source_result:
                    self.flow_map['filtered_to_answer'].append((source_result['row'], dest_row))
                if fingerprint not in self.result_connections: self.result_connections[fingerprint] = {}
                self.result_connections[fingerprint]['used_in_answer'] = True
            
            self.answers.append({
                'question': answer.get('question', 'Unknown'), 'answer': answer.get('answer', 'No answer'),
                'source_file': Path(file_path).name, 'referenced_fingerprints': referenced_fingerprints,
                'row': dest_row
            })

    def _find_original_by_fingerprint(self, fingerprint):
        return next((r for r in self.search_results if r['fingerprint'] == fingerprint), None)

    def _find_filtered_by_fingerprint(self, fingerprint):
        return next((r for r in self.filtered_results if r['fingerprint'] == fingerprint), None)

    def create_visual_flow_excel(self, output_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Visual Data Flow"
        
        regions = {
            'search': {'start_col': 1, 'width': 5, 'title': '1. SEARCH RESULTS'},
            'flow1':  {'col': 6},
            'filtered': {'start_col': 7, 'width': 5, 'title': '2. FILTERED RESULTS'},  
            'flow2':  {'col': 12},
            'answers': {'start_col': 13, 'width': 6, 'title': '3. FINAL ANSWERS'}
        }
        
        self._create_region_headers(ws, regions)
        self._populate_regions(ws, regions)
        self._draw_flow_connectors(ws, regions)
        self._apply_global_formatting(ws, regions)
        
        wb.save(output_path)
        print(f"\n✅ Visual flow report with connectors saved to: {output_path}")

    def _create_region_headers(self, ws, regions):
        header_row = 1
        for region_name, region_info in regions.items():
            if 'title' in region_info:
                start_col, width, title = region_info['start_col'], region_info['width'], region_info['title']
                ws.merge_cells(start_row=header_row, start_column=start_col, end_row=header_row, end_column=start_col + width - 1)
                header_cell = ws.cell(row=header_row, column=start_col, value=title)
                header_cell.font = Font(size=14, bold=True, color="FFFFFF")
                header_cell.alignment = Alignment(horizontal="center", vertical="center")
                fill_colors = {'search': "1565C0", 'filtered': "2E7D32", 'answers': "C62828"}
                header_cell.fill = PatternFill(fill_type="solid", start_color=fill_colors[region_name])
    
    def _populate_regions(self, ws, regions):
        # Helper to set headers
        def set_headers(start_col, headers):
            for i, header in enumerate(headers):
                cell = ws.cell(row=2, column=start_col + i, value=header)
                cell.font = Font(bold=True)
        
        # Helper to fill a row
        def fill_row(row, start_col, data, color):
            for i, value in enumerate(data):
                cell = ws.cell(row=row, column=start_col + i, value=value)
                cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")

        # Populate Search Region
        sr_col = regions['search']['start_col']
        set_headers(sr_col, ['File', 'Pos', 'Fingerprint', 'Title', 'Status'])
        for r in self.search_results:
            fp = r['fingerprint']
            conn = self.result_connections.get(fp, {})
            if conn.get('used_in_answer'): status, color = "Used in Answer", "C8E6C9"
            elif conn.get('filtered'): status, color = "Filtered", "FFF9C4"
            else: status, color = "Stopped", "FFCDD2"
            fill_row(r['row'], sr_col, [r['source_file'], r['position'], fp, r.get('title', '')[:30]+"...", status], color)

        # Populate Filtered Region
        fr_col = regions['filtered']['start_col']
        set_headers(fr_col, ['File', 'Fingerprint', 'Title', 'From Pos', 'Status'])
        for r in self.filtered_results:
            fp = r['fingerprint']
            conn = self.result_connections.get(fp, {})
            if conn.get('used_in_answer'): status, color = "Used in Answer", "C8E6C9"
            else: status, color = "Unused", "FFF9C4"
            fill_row(r['row'], fr_col, [r['source_file'], fp, r.get('title', '')[:25]+"...", r['original_position'], status], color)

        # Populate Answers Region
        ar_col = regions['answers']['start_col']
        set_headers(ar_col, ['File', 'Question', 'Answer', 'Src Count', 'Sources', 'Value'])
        for r in self.answers:
            count = len(r['referenced_fingerprints'])
            color = "C8E6C9" if count > 0 else "FFCDD2"
            value = "High" if count >= 2 else "Low"
            sources_str = ", ".join(r['referenced_fingerprints'])[:30]
            fill_row(r['row'], ar_col, [r['source_file'], r['question'][:20]+"...", r['answer'][:30]+"...", count, sources_str, value], color)

    def _draw_flow_connectors(self, ws, regions):
        flow_font = Font(color="595959", name="Consolas")
        center_align = Alignment(horizontal="center", vertical="center")
        
        flow1_col = regions['flow1']['col']
        for start_row, end_row in self.flow_map['search_to_filtered']:
            self._draw_single_connector(ws, flow1_col, start_row, end_row, flow_font, center_align)
            
        flow2_col = regions['flow2']['col']
        for start_row, end_row in self.flow_map['filtered_to_answer']:
            self._draw_single_connector(ws, flow2_col, start_row, end_row, flow_font, center_align)

    def _draw_single_connector(self, ws, col, start_row, end_row, font, alignment):
        if start_row == end_row: ws.cell(row=start_row, column=col).value = '─►'
        elif start_row < end_row:
            ws.cell(row=start_row, column=col).value = '─╮'
            for r in range(start_row + 1, end_row): ws.cell(row=r, column=col).value = ' │'
            ws.cell(row=end_row, column=col).value = ' └►'
        else: # start_row > end_row
            ws.cell(row=start_row, column=col).value = '─╯'
            for r in range(end_row + 1, start_row): ws.cell(row=r, column=col).value = ' │'
            ws.cell(row=end_row, column=col).value = ' ┌►'
        
        for r in range(min(start_row, end_row), max(start_row, end_row) + 1):
            cell = ws.cell(row=r, column=col)
            cell.font = font
            cell.alignment = alignment

    def _apply_global_formatting(self, ws, regions):
        ws.column_dimensions[get_column_letter(regions['flow1']['col'])].width = 5
        ws.column_dimensions[get_column_letter(regions['flow2']['col'])].width = 5
        # Set other column widths as needed...
        for col_letter in ['A','G','M']: ws.column_dimensions[col_letter].width = 15
        for col_letter in ['D','I','N']: ws.column_dimensions[col_letter].width = 35


# ==============================================================================
#  STEP 2: SETUP DUMMY DATA
#  (This function creates sample JSON files for the reporter to read)
# ==============================================================================

def setup_dummy_json_files():
    """Creates two dummy JSON files with sample data."""
    print("Creating dummy JSON files for demonstration...")

    # Data for the first file
    data1 = {
        "search_results": [
            {"title": "Core AI Concepts", "text": "AI is the simulation..."},
            {"title": "Outdated ML Techniques", "text": "Perceptrons are old..."},
            {"title": "Advanced Neural Networks", "text": "Transformers are key..."}
        ],
        "filtered_search_results": [
            {"title": "Core AI Concepts", "text": "AI is the simulation..."},
            {"title": "Advanced Neural Networks", "text": "Transformers are key..."}
        ],
        "answers_to_questions": [
            {
                "question": "What are the fundamentals of AI?",
                "answer": "The fundamentals of AI revolve around simulating intelligence.",
                "referenced_results": [{"title": "Core AI Concepts", "text": "AI is the simulation..."}]
            }
        ]
    }

    # Data for the second file
    data2 = {
        "search_results": [
            {"title": "AI in Healthcare", "text": "AI helps in diagnostics..."},
            {"title": "Robotics in Surgery", "text": "Da Vinci system is a robot..."}
        ],
        "filtered_search_results": [
            {"title": "AI in Healthcare", "text": "AI helps in diagnostics..."},
            {"title": "Robotics in Surgery", "text": "Da Vinci system is a robot..."}
        ],
        "answers_to_questions": [
            {
                "question": "How is AI used in medicine?",
                "answer": "AI is used for diagnostics and in surgical robotics.",
                "referenced_results": [
                    {"title": "AI in Healthcare", "text": "AI helps in diagnostics..."},
                    {"title": "Robotics in Surgery", "text": "Da Vinci system is a robot..."}
                ]
            }
        ]
    }

    file_names = ["report_data_1.json", "report_data_2.json"]
    all_data = [data1, data2]

    for name, data in zip(file_names, all_data):
        with open(name, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Successfully created: {', '.join(file_names)}")
    return file_names


# ==============================================================================
#  STEP 3: RUN THE REPORTER
#  (This is the main part of the script that executes the logic)
# ==============================================================================

if __name__ == "__main__":
    # 1. Create the dummy data files
    json_files = setup_dummy_json_files()
    
    # 2. Define the name for our output Excel report
    output_excel_file = "visual_data_flow_report.xlsx"
    
    # 3. Instantiate the reporter class
    reporter = VisualDataFlowReporter()
    
    # 4. Load all data from the JSON files and analyze the relationships
    reporter.load_and_analyze_data(json_files)
    
    # 5. Create the final Excel report with visual connectors
    reporter.create_visual_flow_excel(output_excel_file)
    
    # Optional: Clean up the dummy files
    # for file in json_files:
    #     os.remove(file)
    # print("Cleaned up dummy files.")
