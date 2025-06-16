import json
import os
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ==============================================================================
#  STEP 1: DEFINE THE VISUAL DATA FLOW REPORTER CLASS
#  (This is the modified class)
# ==============================================================================

class VisualDataFlowReporter:
    def __init__(self):
        self.search_results = []
        self.filtered_results = []
        self.answers = []
        self.result_connections = {}

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
            doc_id = result.get('doc_id', 'N/A')
            enhanced_result = {
                'doc_id': doc_id, 'source_file': Path(file_path).name,
                'position': idx + 1, 'row': 3 + len(self.search_results),
                **result
            }
            self.search_results.append(enhanced_result)
    
    def _process_filtered_results(self, data, file_path):
        filtered_results = data.get('filtered_search_results', [])
        for result in filtered_results:
            doc_id = result.get('doc_id', 'N/A')
            title = result.get('title') # Get title for matching
            # MODIFIED: Find original using both doc_id and title
            original_result = self._find_original_by_doc_id_and_title(doc_id, title)
            if original_result:
                dest_row = 3 + len(self.filtered_results)
                enhanced_result = {
                    'doc_id': doc_id, 'source_file': Path(file_path).name,
                    'original_position': original_result['position'], 'row': dest_row,
                    **result
                }
                self.filtered_results.append(enhanced_result)
                if doc_id not in self.result_connections: self.result_connections[doc_id] = {}
                self.result_connections[doc_id]['filtered'] = True

    def _process_answers(self, data, file_path):
        answers = data.get('answers_to_questions', [])
        for answer in answers:
            dest_row = 3 + len(self.answers)
            referenced_doc_ids = []
            for ref_result in answer.get('referenced_results', []):
                doc_id = ref_result.get('doc_id', 'N/A')
                title = ref_result.get('title') # Get title for matching
                referenced_doc_ids.append(doc_id)
                # MODIFIED: Find filtered result using both doc_id and title
                source_result = self._find_filtered_by_doc_id_and_title(doc_id, title)
                # Mark connection regardless of whether it was in the filtered list
                if doc_id not in self.result_connections: self.result_connections[doc_id] = {}
                self.result_connections[doc_id]['used_in_answer'] = True
            
            self.answers.append({
                'question': answer.get('question', 'Unknown'), 'answer': answer.get('answer', 'No answer'),
                'source_file': Path(file_path).name, 'referenced_doc_ids': referenced_doc_ids,
                'row': dest_row
            })

    # RENAMED and MODIFIED: Now finds by both doc_id and title.
    def _find_original_by_doc_id_and_title(self, doc_id, title):
        """Finds a result in the initial search list by both doc_id and title."""
        return next((r for r in self.search_results if r['doc_id'] == doc_id and r.get('title') == title), None)

    # RENAMED and MODIFIED: Now finds by both doc_id and title.
    def _find_filtered_by_doc_id_and_title(self, doc_id, title):
        """Finds a result in the filtered list by both doc_id and title."""
        return next((r for r in self.filtered_results if r['doc_id'] == doc_id and r.get('title') == title), None)

    def create_visual_flow_excel(self, output_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data Flow"
        
        # Modified regions to remove the flow connector columns
        regions = {
            'search':   {'start_col': 1, 'width': 5, 'title': '1. SEARCH RESULTS'},
            'filtered': {'start_col': 7, 'width': 5, 'title': '2. FILTERED RESULTS'},  
            'answers':  {'start_col': 13, 'width': 6, 'title': '3. FINAL ANSWERS'}
        }
        
        self._create_region_headers(ws, regions)
        self._populate_regions(ws, regions)
        self._apply_global_formatting(ws, regions)
        
        wb.save(output_path)
        print(f"\n✅ Simplified data flow report saved to: {output_path}")

    def _create_region_headers(self, ws, regions):
        header_row = 1
        ws.row_dimensions[header_row].height = 25
        for region_name, region_info in regions.items():
            if 'title' in region_info:
                start_col, width, title = region_info['start_col'], region_info['width'], region_info['title']
                # Add a blank column as a separator
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
        set_headers(sr_col, ['File', 'Pos', 'Doc ID', 'Title', 'Status'])
        for r in self.search_results:
            doc_id = r['doc_id']
            conn = self.result_connections.get(doc_id, {})
            if conn.get('used_in_answer'): status, color = "Used in Answer", "C8E6C9"
            elif conn.get('filtered'): status, color = "Filtered", "FFF9C4"
            else: status, color = "Stopped", "FFCDD2"
            fill_row(r['row'], sr_col, [r['source_file'], r['position'], doc_id, r.get('title', '')[:30]+"...", status], color)

        # Populate Filtered Region
        fr_col = regions['filtered']['start_col']
        set_headers(fr_col, ['File', 'Doc ID', 'Title', 'From Pos', 'Status'])
        for r in self.filtered_results:
            doc_id = r['doc_id']
            conn = self.result_connections.get(doc_id, {})
            if conn.get('used_in_answer'): status, color = "Used in Answer", "C8E6C9"
            else: status, color = "Unused", "FFF9C4"
            fill_row(r['row'], fr_col, [r['source_file'], doc_id, r.get('title', '')[:25]+"...", r['original_position'], status], color)

        # Populate Answers Region
        ar_col = regions['answers']['start_col']
        set_headers(ar_col, ['File', 'Question', 'Answer', 'Src Count', 'Source IDs', 'Value'])
        for r in self.answers:
            count = len(r['referenced_doc_ids'])
            color = "C8E6C9" if count > 0 else "FFCDD2"
            value = "High" if count >= 2 else "Low"
            sources_str = ", ".join(r['referenced_doc_ids'])[:30]
            fill_row(r['row'], ar_col, [r['source_file'], r['question'][:20]+"...", r['answer'][:30]+"...", count, sources_str, value], color)

    def _apply_global_formatting(self, ws, regions):
        # Add blank separator columns
        ws.column_dimensions[get_column_letter(regions['filtered']['start_col'] - 1)].width = 3
        ws.column_dimensions[get_column_letter(regions['answers']['start_col'] - 1)].width = 3

        # Set content column widths
        # Search Region Columns (A-E)
        ws.column_dimensions['A'].width = 15 # File
        ws.column_dimensions['C'].width = 15 # Doc ID
        ws.column_dimensions['D'].width = 35 # Title
        # Filtered Region Columns (G-K)
        ws.column_dimensions['G'].width = 15 # File
        ws.column_dimensions['H'].width = 15 # Doc ID
        ws.column_dimensions['I'].width = 35 # Title
        # Answer Region Columns (M-R)
        ws.column_dimensions['M'].width = 15 # File
        ws.column_dimensions['N'].width = 35 # Question
        ws.column_dimensions['O'].width = 35 # Answer


# ==============================================================================
#  STEP 2: SETUP DUMMY DATA
#  (This function is updated to include `doc_id` in the sample data)
# ==============================================================================

def setup_dummy_json_files():
    """Creates two dummy JSON files with sample data including doc_id."""
    print("Creating dummy JSON files for demonstration...")

    # Data for the first file
    data1 = {
        "search_results": [
            {"doc_id": "doc-ai-1", "title": "Core AI Concepts", "text": "AI is the simulation..."},
            {"doc_id": "doc-ai-2", "title": "Outdated ML Techniques", "text": "Perceptrons are old..."},
            {"doc_id": "doc-ai-3", "title": "Advanced Neural Networks", "text": "Transformers are key..."}
        ],
        "filtered_search_results": [
            {"doc_id": "doc-ai-1", "title": "Core AI Concepts", "text": "AI is the simulation..."},
            {"doc_id": "doc-ai-3", "title": "Advanced Neural Networks", "text": "Transformers are key..."}
        ],
        "answers_to_questions": [
            {
                "question": "What are the fundamentals of AI?",
                "answer": "The fundamentals of AI revolve around simulating intelligence.",
                "referenced_results": [{"doc_id": "doc-ai-1", "title": "Core AI Concepts", "text": "AI is the simulation..."}]
            }
        ]
    }

    # Data for the second file
    data2 = {
        "search_results": [
            {"doc_id": "doc-med-1", "title": "AI in Healthcare", "text": "AI helps in diagnostics..."},
            {"doc_id": "doc-med-2", "title": "Robotics in Surgery", "text": "Da Vinci system is a robot..."}
        ],
        "filtered_search_results": [
            {"doc_id": "doc-med-1", "title": "AI in Healthcare", "text": "AI helps in diagnostics..."},
            {"doc_id": "doc-med-2", "title": "Robotics in Surgery", "text": "Da Vinci system is a robot..."}
        ],
        "answers_to_questions": [
            {
                "question": "How is AI used in medicine?",
                "answer": "AI is used for diagnostics and in surgical robotics.",
                "referenced_results": [
                    {"doc_id": "doc-med-1", "title": "AI in Healthcare", "text": "AI helps in diagnostics..."},
                    {"doc_id": "doc-med-2", "title": "Robotics in Surgery", "text": "Da Vinci system is a robot..."}
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
    output_excel_file = "simplified_data_flow_report.xlsx"
    
    # 3. Instantiate the reporter class
    reporter = VisualDataFlowReporter()
    
    # 4. Load all data from the JSON files and analyze the relationships
    reporter.load_and_analyze_data(json_files)
    
    # 5. Create the final Excel report with the simplified view
    reporter.create_visual_flow_excel(output_excel_file)
    
    # Optional: Clean up the dummy files
    # for file in json_files:
    #     os.remove(file)
    # print("Cleaned up dummy files.")
