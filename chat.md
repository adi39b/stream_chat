<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I need to build a dashboard on top of data coming from databricks. It has to connect to Databricks using python sql connector. The dashboard frontend will need to seamlessly connect and show real time graphs about comprehensive breakdowns of Direct costs and indirect costs in a large organisation. Think of detailed mappings and customisability across trees of sub organization structures, trees of financial element structures and so on. The customisable data modeling is important

Building a comprehensive real-time dashboard for organizational cost analysis requires a multi-layered architecture that handles data connectivity, real-time streaming, and sophisticated visualization. Here's a detailed implementation guide for your Databricks-powered dashboard.

## Architecture Overview

The optimal architecture combines Databricks for data processing, FastAPI for real-time API services, and React for dynamic frontend visualization [^1_1][^1_2]. This setup leverages Databricks' native Python SQL connector for data retrieval, WebSockets for real-time updates, and D3.js for hierarchical tree visualizations [^1_3][^1_4][^1_5].

## Databricks Connection and Data Layer

### Python SQL Connector Setup

The Databricks SQL Connector for Python version 3.0.0+ supports native parameterized queries and follows PEP 249 standards [^1_3][^1_6]. Install the connector with Arrow support for optimal performance:

```python
pip install databricks-sql-connector[pyarrow]
```

**Connection Configuration:**

```python
from databricks import sql
import os

connection = sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_TOKEN")
)
```


### Data Model Design

Design a hierarchical cost breakdown structure (CBS) that supports both organizational and financial hierarchies [^1_7][^1_8]. Key tables should include:

**Organizational Structure:**

- `organizations` (id, name, parent_id, level, cost_center)
- `cost_allocations` (org_id, period, direct_costs, indirect_costs)

**Financial Structure:**

- `cost_categories` (id, name, parent_id, category_type)
- `cost_transactions` (id, org_id, category_id, amount, transaction_date, cost_type)

This structure enables flexible drilling down through organizational hierarchies while maintaining financial categorization [^1_9][^1_10].

## Backend API Development

### FastAPI Real-time Service

Implement a FastAPI backend with WebSocket support for real-time data streaming [^1_11][^1_12]. The service handles both REST endpoints for initial data loading and WebSocket connections for live updates:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from datetime import datetime

class DashboardManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        
    async def broadcast_cost_updates(self, data):
        for connection in self.connections:
            try:
                await connection.send_text(json.dumps(data))
            except:
                self.connections.remove(connection)

app = FastAPI()
manager = DashboardManager()

@app.websocket("/ws/costs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.connections.append(websocket)
```


### Data Processing Pipeline

Implement real-time mode using Structured Streaming for sub-second latency [^1_13]. Create aggregation queries that calculate direct and indirect costs across organizational hierarchies:

```python
def get_cost_breakdown(org_id, period):
    query = """
    WITH RECURSIVE org_tree AS (
        SELECT id, name, parent_id, 0 as level
        FROM organizations WHERE id = ?
        UNION ALL
        SELECT o.id, o.name, o.parent_id, ot.level + 1
        FROM organizations o
        JOIN org_tree ot ON o.parent_id = ot.id
    )
    SELECT 
        ot.id, ot.name, ot.level,
        SUM(ca.direct_costs) as direct_costs,
        SUM(ca.indirect_costs) as indirect_costs
    FROM org_tree ot
    LEFT JOIN cost_allocations ca ON ot.id = ca.org_id
    WHERE ca.period = ?
    GROUP BY ot.id, ot.name, ot.level
    """
    return execute_query(query, [org_id, period])
```


## Frontend Implementation

### React Dashboard Architecture

Build the frontend using React with real-time WebSocket integration [^1_4][^1_2]. Use hooks for WebSocket management and state handling:

```javascript
const useWebSocket = (url) => {
    const [data, setData] = useState(null);
    const ws = useRef(null);
    
    useEffect(() => {
        ws.current = new WebSocket(url);
        ws.current.onmessage = (event) => {
            const newData = JSON.parse(event.data);
            setData(newData);
        };
        
        return () => ws.current.close();
    }, [url]);
    
    return data;
};
```


### Hierarchical Visualization Components

Implement tree visualizations using D3.js for organizational structures [^1_5][^1_14]. Create collapsible, interactive organization charts that display cost allocations:

**D3 Tree Layout:**

```javascript
import * as d3 from 'd3';

const createOrgTree = (data, containerId) => {
    const hierarchy = d3.hierarchy(data);
    const treeLayout = d3.tree().size([800, 600]);
    
    treeLayout(hierarchy);
    
    const svg = d3.select(`#${containerId}`)
        .append('svg')
        .attr('width', 900)
        .attr('height', 700);
        
    // Add nodes with cost information
    const nodes = svg.selectAll('.node')
        .data(hierarchy.descendants())
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.x},${d.y})`);
        
    nodes.append('rect')
        .attr('width', 120)
        .attr('height', 60)
        .style('fill', d => costColorScale(d.data.totalCosts));
        
    nodes.append('text')
        .text(d => `${d.data.name}\n$${d.data.totalCosts}`);
};
```

**Treemap for Cost Visualization:**
Implement treemaps for hierarchical cost analysis using nested rectangles where size represents cost magnitude [^1_15][^1_9]:

```javascript
const createCostTreemap = (data) => {
    const treemap = d3.treemap()
        .size([800, 600])
        .padding(2);
        
    const root = d3.hierarchy(data)
        .sum(d => d.totalCosts)
        .sort((a, b) => b.totalCosts - a.totalCosts);
        
    treemap(root);
    
    // Render rectangles with cost data
    const cells = svg.selectAll('.cell')
        .data(root.leaves())
        .enter()
        .append('g')
        .attr('class', 'cell');
};
```


## Customizable Data Modeling

### Dynamic Schema Management

Implement a flexible metadata system that allows users to define custom organizational and financial hierarchies [^1_16]. Create configuration tables:

```sql
CREATE TABLE custom_hierarchies (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    hierarchy_type ENUM('organizational', 'financial'),
    parent_mapping TEXT, -- JSON configuration
    cost_allocation_rules TEXT -- JSON rules
);
```


### Interactive Dashboard Configuration

Build configuration panels that allow users to:

- Define custom organizational structures
- Map cost centers to departments
- Configure direct vs indirect cost classifications
- Set up real-time update intervals [^1_17][^1_18]

**Configuration Interface:**

```javascript
const HierarchyConfig = () => {
    const [hierarchyConfig, setHierarchyConfig] = useState({
        orgStructure: [],
        costCategories: [],
        allocationRules: {}
    });
    
    const updateHierarchy = (newConfig) => {
        setHierarchyConfig(newConfig);
        // Send to backend for persistence
        api.updateHierarchyConfig(newConfig);
    };
    
    return (
        <ConfigPanel>
            <TreeEditor 
                data={hierarchyConfig.orgStructure}
                onChange={updateHierarchy}
            />
            <CostRuleEditor 
                rules={hierarchyConfig.allocationRules}
                onChange={updateHierarchy}
            />
        </ConfigPanel>
    );
};
```


## Performance Optimization

### Real-time Data Management

Implement efficient data streaming with backpressure control and client-side windowing [^1_2]. Use data buffering to prevent UI overload:

```javascript
const useBufferedData = (rawData, bufferSize = 100) => {
    const [buffer, setBuffer] = useState([]);
    
    useEffect(() => {
        if (rawData) {
            setBuffer(prev => 
                [...prev, rawData].slice(-bufferSize)
            );
        }
    }, [rawData, bufferSize]);
    
    return buffer;
};
```


### Caching Strategy

Implement multi-level caching with Databricks SQL warehouses providing up to 25% performance improvement [^1_19]. Use Redis for frequently accessed hierarchical data and browser storage for user preferences.

This architecture provides a scalable foundation for comprehensive organizational cost analysis with real-time updates, customizable hierarchies, and interactive visualization capabilities. The modular design allows for easy extension and modification as organizational requirements evolve.
<span style="display:none">[^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">â‚</div>

[^1_1]: https://www.databricks.com/blog/building-databricks-apps-react-and-mosaic-ai-agents-enterprise-chat-solutions

[^1_2]: https://makersden.io/blog/reactjs-dev-for-real-time-analytics-dashboards

[^1_3]: https://docs.databricks.com/aws/en/dev-tools/python-sql-connector

[^1_4]: https://innovationm.com/blog/react-websockets/

[^1_5]: https://d3indepth.com/hierarchies/

[^1_6]: https://learn.microsoft.com/en-us/azure/databricks/dev-tools/python-sql-connector

[^1_7]: https://www.projectmanager.com/blog/cost-breakdown-structure

[^1_8]: https://cleopatraenterprise.com/blog/cost-breakdown-structure-cbs/

[^1_9]: https://curatepartners.com/blogs/skills-tools-platforms/tree-maps-visualizing-hierarchical-data-for-better-business-insights-curate-consulting/

[^1_10]: https://www.saviom.com/blog/cost-breakdown-structure/

[^1_11]: https://krython.com/tutorial/python/fastapi-websockets-real-time/

[^1_12]: https://testdriven.io/blog/fastapi-postgres-websockets/

[^1_13]: https://community.databricks.com/t5/product-platform-updates/what-s-new-in-databricks-july-august-2025/ba-p/130308

[^1_14]: https://d3js.org/d3-hierarchy

[^1_15]: https://help.astrato.io/en/articles/10118326-treemap-visualization

[^1_16]: https://learn.microsoft.com/en-us/dynamics365/sales/hierarchy-visualization

[^1_17]: https://www.openmindt.com/knowledge/example-of-a-cost-control-management-dashboard/

[^1_18]: https://aws.amazon.com/blogs/architecture/building-a-showback-dashboard-for-cost-visibility-with-serverless-architectures/

[^1_19]: https://learn.microsoft.com/en-us/azure/databricks/sql/release-notes/2025

[^1_20]: https://docs.databricks.com/aws/en/dashboards/

[^1_21]: https://www.cdata.com/blog/sqlserver-databricks-2025

[^1_22]: https://pypi.org/project/databricks-sql-connector/

[^1_23]: https://www.chaosgenius.io/blog/databricks-dashboard/

[^1_24]: https://www.nngroup.com/articles/treemaps/

[^1_25]: https://www.youtube.com/watch?v=fzlSd_rVzf8

[^1_26]: https://www.interaction-design.org/literature/topics/hierarchical-data-visualization

[^1_27]: https://www.databricks.com/resources/demos/videos/streaming-data-power-bi-azure-databricks

[^1_28]: https://dev.to/encore/how-to-build-a-real-time-dashboard-with-encorets-and-react-ii9

[^1_29]: https://www.youtube.com/watch?v=6a3Dz8gwjdg

[^1_30]: https://blog.pixelfreestudio.com/how-to-implement-real-time-data-handling-with-react/

[^1_31]: https://stackoverflow.com/questions/30926539/organization-chart-tree-online-dynamic-collapsible-pictures-in-d3

[^1_32]: https://seenode.com/blog/websockets-with-fastapi-real-time-apps-tutorial/

[^1_33]: https://python.plainenglish.io/rebuilding-my-real-time-dashboard-with-react-39e0c92b2359

[^1_34]: https://d3js.org/d3-hierarchy/tree

[^1_35]: https://testdriven.io/blog/fastapi-mongo-websockets/

[^1_36]: https://onyxdev.net/snippets-item/organizational-chart-with-d3-js-expandable-zoomable-and-fully-initialized/

[^1_37]: https://fastapi.tiangolo.com/advanced/websockets/

[^1_38]: https://d3js.org

[^1_39]: https://leapcell.io/blog/real-time-communication-in-python-with-websockets-and-fastapi


---

# There needs to be a cost calculator function as well, which needs to be customisable. Build the frontend with plotly dash, that would be terrific. It should have multiple pages with relevant sections and names, should be a lot more detailed than this.

Building a comprehensive multi-page cost analysis dashboard with Plotly Dash requires a sophisticated architecture that handles real-time data streaming, customizable cost calculations, and hierarchical organizational structures. Here's a detailed implementation guide for your Databricks-powered dashboard.

## Project Structure and Architecture

The dashboard follows a modular multi-page structure using Dash Pages with real-time Databricks integration [^2_1][^2_2]. The architecture separates data processing, cost calculation logic, and visualization components for maximum flexibility [^2_3][^2_4].

**Project Structure:**

```
cost_dashboard/
â”œâ”€â”€ app.py                 # Main application entry point
â”œâ”€â”€ pages/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ home.py           # Overview dashboard
â”‚   â”œâ”€â”€ cost_analysis.py  # Detailed cost breakdowns  
â”‚   â”œâ”€â”€ calculator.py     # Interactive cost calculator
â”‚   â”œâ”€â”€ hierarchy.py      # Organizational structure management
â”‚   â””â”€â”€ settings.py       # Configuration and admin panel
â”œâ”€â”€ components/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ databricks_client.py    # Database connectivity
â”‚   â”œâ”€â”€ cost_calculator.py      # Core calculation engine
â”‚   â”œâ”€â”€ real_time_updates.py    # WebSocket/interval handlers
â”‚   â””â”€â”€ visualization_utils.py  # Reusable chart components
â”œâ”€â”€ assets/
â”‚   â””â”€â”€ custom.css        # Dashboard styling
â””â”€â”€ config/
    â””â”€â”€ settings.py       # Configuration management
```


## Databricks Integration Layer

### Enhanced SQL Connector Setup

The Databricks SQL connector provides native integration with parameterized queries and real-time streaming capabilities [^2_5][^2_6]:

```python
# components/databricks_client.py
import os
from databricks import sql
import pandas as pd
from sqlalchemy import create_engine
import threading
import time

class DatabricksClient:
    def __init__(self):
        self.connection_params = {
            'server_hostname': os.getenv('DATABRICKS_SERVER_HOSTNAME'),
            'http_path': os.getenv('DATABRICKS_HTTP_PATH'),
            'access_token': os.getenv('DATABRICKS_TOKEN')
        }
        self.connection = None
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Establish connection with retry logic"""
        self.connection = sql.connect(**self.connection_params)
        
        # SQLAlchemy engine for pandas integration
        connection_string = f"databricks://token:{self.connection_params['access_token']}@{self.connection_params['server_hostname']}:443/{self.connection_params['http_path']}"
        self.engine = create_engine(connection_string)
    
    def execute_query(self, query, params=None):
        """Execute parameterized query with error handling"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or [])
                return cursor.fetchall()
        except Exception as e:
            self._reconnect()
            raise e
    
    def get_cost_hierarchy_data(self, org_id=None, date_range=None):
        """Retrieve hierarchical cost data with optional filtering"""
        query = """
        WITH RECURSIVE org_hierarchy AS (
            SELECT 
                id, name, parent_id, level, cost_center_code,
                direct_cost_rate, indirect_cost_rate
            FROM organizations
            WHERE (? IS NULL OR id = ?)
            UNION ALL
            SELECT 
                o.id, o.name, o.parent_id, oh.level + 1, o.cost_center_code,
                o.direct_cost_rate, o.indirect_cost_rate
            FROM organizations o
            INNER JOIN org_hierarchy oh ON o.parent_id = oh.id
        ),
        cost_aggregation AS (
            SELECT 
                oh.id, oh.name, oh.level, oh.cost_center_code,
                oh.direct_cost_rate, oh.indirect_cost_rate,
                COALESCE(SUM(ca.direct_costs), 0) as total_direct_costs,
                COALESCE(SUM(ca.indirect_costs), 0) as total_indirect_costs,
                COALESCE(SUM(ca.allocated_costs), 0) as total_allocated_costs,
                COUNT(DISTINCT ca.transaction_id) as transaction_count
            FROM org_hierarchy oh
            LEFT JOIN cost_allocations ca ON oh.id = ca.org_id
            WHERE (? IS NULL OR ca.transaction_date >= ?)
              AND (? IS NULL OR ca.transaction_date <= ?)
            GROUP BY oh.id, oh.name, oh.level, oh.cost_center_code,
                     oh.direct_cost_rate, oh.indirect_cost_rate
        )
        SELECT * FROM cost_aggregation
        ORDER BY level, name
        """
        
        start_date = date_range[^2_0] if date_range else None
        end_date = date_range[^2_1] if date_range else None
        
        return pd.DataFrame(
            self.execute_query(query, [org_id, org_id, start_date, start_date, end_date, end_date])
        )

# Real-time data streaming
class RealTimeDataStream:
    def __init__(self, db_client):
        self.db_client = db_client
        self.subscribers = []
        self.running = False
        
    def subscribe(self, callback):
        """Register callback for real-time updates"""
        self.subscribers.append(callback)
        
    def start_streaming(self, interval=30):
        """Start real-time data streaming"""
        self.running = True
        
        def stream_worker():
            while self.running:
                try:
                    # Fetch latest cost data
                    latest_data = self.db_client.get_latest_cost_updates()
                    
                    # Notify all subscribers
                    for callback in self.subscribers:
                        callback(latest_data)
                        
                except Exception as e:
                    print(f"Streaming error: {e}")
                    
                time.sleep(interval)
                
        threading.Thread(target=stream_worker, daemon=True).start()
```


## Advanced Cost Calculator Engine

### Flexible Cost Calculation Framework

The cost calculator supports multiple allocation methods and customizable rules [^2_7][^2_8]:

```python
# components/cost_calculator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class AllocationMethod(Enum):
    DIRECT = "direct"
    STEP_DOWN = "step_down"
    RECIPROCAL = "reciprocal"
    ACTIVITY_BASED = "activity_based"
    USAGE_BASED = "usage_based"

@dataclass
class CostRule:
    name: str
    allocation_method: AllocationMethod
    driver_type: str  # 'headcount', 'revenue', 'square_footage', 'cpu_hours', etc.
    allocation_percentage: Optional[float] = None
    minimum_allocation: Optional[float] = None
    maximum_allocation: Optional[float] = None
    escalation_rate: float = 0.0
    effective_date: Optional[str] = None

class CostCalculatorEngine:
    def __init__(self):
        self.allocation_rules: Dict[str, List[CostRule]] = {}
        self.cost_pools: Dict[str, float] = {}
        self.allocation_drivers: Dict[str, Dict] = {}
        
    def register_cost_rule(self, cost_center: str, rule: CostRule):
        """Register allocation rule for a cost center"""
        if cost_center not in self.allocation_rules:
            self.allocation_rules[cost_center] = []
        self.allocation_rules[cost_center].append(rule)
    
    def set_allocation_drivers(self, drivers: Dict[str, Dict]):
        """Set driver data for allocations (headcount, revenue, etc.)"""
        self.allocation_drivers = drivers
    
    def calculate_direct_costs(self, org_data: pd.DataFrame, 
                             cost_pools: Dict[str, float]) -> pd.DataFrame:
        """Calculate direct cost allocations"""
        results = org_data.copy()
        
        for org_id, row in results.iterrows():
            direct_costs = 0
            
            # Apply direct cost rules
            if row['id'] in self.allocation_rules:
                for rule in self.allocation_rules[row['id']]:
                    if rule.allocation_method == AllocationMethod.DIRECT:
                        cost_amount = cost_pools.get(rule.name, 0)
                        
                        if rule.allocation_percentage:
                            direct_costs += cost_amount * rule.allocation_percentage
                        else:
                            direct_costs += cost_amount
            
            results.loc[org_id, 'calculated_direct_costs'] = direct_costs
            
        return results
    
    def calculate_step_down_allocation(self, org_hierarchy: pd.DataFrame,
                                     shared_costs: Dict[str, float]) -> pd.DataFrame:
        """Implement step-down cost allocation method"""
        results = org_hierarchy.copy()
        
        # Sort by hierarchy level (top-down allocation)
        sorted_orgs = results.sort_values('level')
        
        for cost_pool, total_cost in shared_costs.items():
            remaining_cost = total_cost
            
            for _, org in sorted_orgs.iterrows():
                if org['id'] in self.allocation_rules:
                    applicable_rules = [r for r in self.allocation_rules[org['id']] 
                                      if r.name == cost_pool and 
                                      r.allocation_method == AllocationMethod.STEP_DOWN]
                    
                    for rule in applicable_rules:
                        # Calculate allocation based on driver
                        driver_value = self._get_driver_value(org['id'], rule.driver_type)
                        total_driver_value = self._get_total_driver_value(
                            sorted_orgs[sorted_orgs['level'] >= org['level']], 
                            rule.driver_type
                        )
                        
                        if total_driver_value > 0:
                            allocation_ratio = driver_value / total_driver_value
                            allocated_amount = remaining_cost * allocation_ratio
                            
                            # Apply constraints
                            if rule.minimum_allocation:
                                allocated_amount = max(allocated_amount, rule.minimum_allocation)
                            if rule.maximum_allocation:
                                allocated_amount = min(allocated_amount, rule.maximum_allocation)
                            
                            results.loc[results['id'] == org['id'], 'allocated_costs'] += allocated_amount
                            remaining_cost -= allocated_amount
        
        return results
    
    def calculate_activity_based_costs(self, activities: pd.DataFrame,
                                     cost_drivers: Dict[str, float]) -> pd.DataFrame:
        """Calculate activity-based costing allocations"""
        results = activities.copy()
        
        # Calculate cost per activity driver
        for activity, cost_pool in cost_drivers.items():
            total_activity_volume = results[f'{activity}_volume'].sum()
            
            if total_activity_volume > 0:
                cost_per_unit = cost_pool / total_activity_volume
                results[f'{activity}_allocated_cost'] = (
                    results[f'{activity}_volume'] * cost_per_unit
                )
        
        return results
    
    def calculate_usage_based_allocation(self, usage_data: pd.DataFrame,
                                       shared_resources: Dict[str, float]) -> pd.DataFrame:
        """Calculate usage-based cost allocation"""
        results = usage_data.copy()
        
        for resource, total_cost in shared_resources.items():
            usage_column = f'{resource}_usage'
            
            if usage_column in results.columns:
                total_usage = results[usage_column].sum()
                
                if total_usage > 0:
                    results[f'{resource}_allocated_cost'] = (
                        (results[usage_column] / total_usage) * total_cost
                    )
        
        return results
    
    def _get_driver_value(self, org_id: str, driver_type: str) -> float:
        """Get allocation driver value for specific organization"""
        return self.allocation_drivers.get(org_id, {}).get(driver_type, 0)
    
    def _get_total_driver_value(self, orgs: pd.DataFrame, driver_type: str) -> float:
        """Calculate total driver value across organizations"""
        total = 0
        for _, org in orgs.iterrows():
            total += self._get_driver_value(org['id'], driver_type)
        return total
    
    def run_comprehensive_allocation(self, org_data: pd.DataFrame,
                                   cost_inputs: Dict) -> Dict[str, pd.DataFrame]:
        """Run complete cost allocation using all configured methods"""
        results = {}
        
        # Direct cost allocation
        results['direct_costs'] = self.calculate_direct_costs(
            org_data, cost_inputs.get('direct_cost_pools', {})
        )
        
        # Step-down allocation
        results['step_down'] = self.calculate_step_down_allocation(
            org_data, cost_inputs.get('shared_costs', {})
        )
        
        # Activity-based costing
        if 'activities' in cost_inputs:
            results['abc_costs'] = self.calculate_activity_based_costs(
                cost_inputs['activities'], cost_inputs.get('cost_drivers', {})
            )
        
        # Usage-based allocation
        if 'usage_data' in cost_inputs:
            results['usage_costs'] = self.calculate_usage_based_allocation(
                cost_inputs['usage_data'], cost_inputs.get('shared_resources', {})
            )
        
        return results

# Customizable calculator interface
class CustomCostCalculator:
    def __init__(self, calculator_engine: CostCalculatorEngine):
        self.engine = calculator_engine
        self.custom_formulas: Dict[str, Callable] = {}
        
    def add_custom_formula(self, name: str, formula: Callable):
        """Add user-defined cost calculation formula"""
        self.custom_formulas[name] = formula
    
    def calculate_with_custom_rules(self, data: pd.DataFrame, 
                                  formula_name: str, **kwargs) -> pd.DataFrame:
        """Apply custom calculation formula"""
        if formula_name in self.custom_formulas:
            return self.custom_formulas[formula_name](data, **kwargs)
        else:
            raise ValueError(f"Custom formula '{formula_name}' not found")
```


## Multi-Page Dashboard Implementation

### Main Application with Navigation

The main app implements Dash Pages for seamless navigation between different dashboard sections [^2_1][^2_9]:

```python
# app.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from components.databricks_client import DatabricksClient, RealTimeDataStream
from components.cost_calculator import CostCalculatorEngine, CustomCostCalculator

# Initialize core components
db_client = DatabricksClient()
cost_engine = CostCalculatorEngine()
custom_calculator = CustomCostCalculator(cost_engine)
data_stream = RealTimeDataStream(db_client)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__, 
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Global navigation component
def create_navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Cost Analysis", href="/cost-analysis", active="exact")),
            dbc.NavItem(dbc.NavLink("Calculator", href="/calculator", active="exact")),
            dbc.NavItem(dbc.NavLink("Hierarchy", href="/hierarchy", active="exact")),
            dbc.NavItem(dbc.NavLink("Settings", href="/settings", active="exact")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Export Data", href="#"),
                    dbc.DropdownMenuItem("Import Rules", href="#"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Help", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand="Enterprise Cost Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        fluid=True,
    )

# Main app layout
app.layout = dbc.Container([
    create_navbar(),
    html.Br(),
    
    # Real-time status indicator
    dbc.Alert(
        [
            html.I(className="fas fa-circle me-2"),
            html.Span("Connected to Databricks", id="connection-status")
        ],
        color="success",
        className="d-flex align-items-center",
        style={"margin-bottom": "20px"}
    ),
    
    # Page content container
    dash.page_container,
    
    # Real-time update components
    dcc.Interval(
        id='realtime-interval',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    ),
    
    # Store components for shared data
    dcc.Store(id='org-hierarchy-store'),
    dcc.Store(id='cost-rules-store'),
    dcc.Store(id='realtime-data-store'),
    
], fluid=True)

if __name__ == '__main__':
    # Start real-time data streaming
    data_stream.start_streaming()
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```


### Home Dashboard Page

The home page provides executive-level overview with real-time metrics [^2_10][^2_11]:

```python
# pages/home.py
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

dash.register_page(__name__, path='/', name='Dashboard Overview')

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("ðŸ“Š Enterprise Cost Dashboard", className="text-center mb-4"),
                html.P("Real-time organizational cost analysis and allocation tracking", 
                       className="text-center text-muted lead"),
                html.Hr()
            ])
        ]),
        
        # Key Performance Indicators
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(id="total-costs", className="text-primary mb-0"),
                        html.P("Total Costs (YTD)", className="text-muted small"),
                        html.P(id="cost-variance", className="mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(id="direct-costs", className="text-success mb-0"),
                        html.P("Direct Costs", className="text-muted small"),
                        html.P(id="direct-percentage", className="mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(id="indirect-costs", className="text-warning mb-0"),
                        html.P("Indirect Costs", className="text-muted small"),
                        html.P(id="indirect-percentage", className="mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(id="cost-centers", className="text-info mb-0"),
                        html.P("Active Cost Centers", className="text-muted small"),
                        html.P(id="efficiency-score", className="mb-0")
                    ])
                ])
            ], width=3),
        ], className="mb-4"),
        
        # Main visualization area
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("ðŸ“ˆ Cost Trends", className="mb-0"),
                        dbc.ButtonGroup([
                            dbc.Button("7D", id="btn-7d", size="sm", outline=True),
                            dbc.Button("30D", id="btn-30d", size="sm", outline=True),
                            dbc.Button("90D", id="btn-90d", size="sm", outline=True, active=True),
                            dbc.Button("YTD", id="btn-ytd", size="sm", outline=True),
                        ], size="sm")
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        dcc.Graph(id="cost-trends-chart", style={'height': '400px'})
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ðŸ¢ Organizational Breakdown"),
                    dbc.CardBody([
                        dcc.Graph(id="org-breakdown-chart", style={'height': '400px'})
                    ])
                ])
            ], width=4),
        ], className="mb-4"),
        
        # Secondary charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ðŸ’° Cost Allocation Waterfall"),
                    dbc.CardBody([
                        dcc.Graph(id="waterfall-chart", style={'height': '300px'})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ðŸ“Š Top Cost Drivers"),
                    dbc.CardBody([
                        dcc.Graph(id="cost-drivers-chart", style={'height': '300px'})
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        
        # Real-time activity feed
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ðŸ”„ Recent Cost Activities"),
                    dbc.CardBody([
                        html.Div(id="activity-feed", style={'height': '200px', 'overflow-y': 'auto'})
                    ])
                ])
            ])
        ])
    ], fluid=True)

# Callbacks for real-time updates
@callback(
    [Output('total-costs', 'children'),
     Output('direct-costs', 'children'),
     Output('indirect-costs', 'children'),
     Output('cost-centers', 'children'),
     Output('cost-variance', 'children'),
     Output('direct-percentage', 'children'),
     Output('indirect-percentage', 'children'),
     Output('efficiency-score', 'children')],
    [Input('realtime-interval', 'n_intervals')],
    [State('org-hierarchy-store', 'data')]
)
def update_kpis(n_intervals, org_data):
    if not org_data:
        return ["$0"] * 4 + [""] * 4
    
    df = pd.DataFrame(org_data)
    
    total_costs = df['total_direct_costs'].sum() + df['total_indirect_costs'].sum()
    direct_costs = df['total_direct_costs'].sum()
    indirect_costs = df['total_indirect_costs'].sum()
    cost_centers = len(df)
    
    # Calculate variance and percentages
    cost_variance = f"â†‘ 5.2% vs last month"  # This would be calculated from historical data
    direct_percentage = f"{(direct_costs/total_costs)*100:.1f}% of total"
    indirect_percentage = f"{(indirect_costs/total_costs)*100:.1f}% of total"
    efficiency_score = f"Efficiency: 94.2%"
    
    return [
        f"${total_costs:,.0f}",
        f"${direct_costs:,.0f}",
        f"${indirect_costs:,.0f}",
        f"{cost_centers:,}",
        cost_variance,
        direct_percentage,
        indirect_percentage,
        efficiency_score
    ]

@callback(
    Output('cost-trends-chart', 'figure'),
    [Input('realtime-interval', 'n_intervals'),
     Input('btn-7d', 'n_clicks'),
     Input('btn-30d', 'n_clicks'),
     Input('btn-90d', 'n_clicks'),
     Input('btn-ytd', 'n_clicks')],
    [State('realtime-data-store', 'data')]
)
def update_cost_trends(n_intervals, btn_7d, btn_30d, btn_90d, btn_ytd, realtime_data):
    # Determine time range based on button clicks
    ctx = dash.callback_context
    if not ctx.triggered:
        period = "90D"
    else:
        button_id = ctx.triggered[^2_0]['prop_id'].split('.')[^2_0]
        period = button_id.split('-')[^2_1] if button_id.startswith('btn-') else "90D"
    
    # Generate sample trend data (replace with actual Databricks query)
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    # Simulate cost trend data
    np.random.seed(42)
    direct_costs = 50000 + np.cumsum(np.random.normal(100, 500, len(dates)))
    indirect_costs = 30000 + np.cumsum(np.random.normal(50, 300, len(dates)))
    
    df_trends = pd.DataFrame({
        'date': dates,
        'direct_costs': direct_costs,
        'indirect_costs': indirect_costs,
        'total_costs': direct_costs + indirect_costs
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add cost lines
    fig.add_trace(
        go.Scatter(x=df_trends['date'], y=df_trends['direct_costs'],
                  name='Direct Costs', line=dict(color='#28a745')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df_trends['date'], y=df_trends['indirect_costs'],
                  name='Indirect Costs', line=dict(color='#ffc107')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df_trends['date'], y=df_trends['total_costs'],
                  name='Total Costs', line=dict(color='#007bff', width=3)),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="Cost Trends Over Time",
        xaxis_title="Date",
        template="plotly_white",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Direct & Indirect Costs ($)", secondary_y=False)
    fig.update_yaxes(title_text="Total Costs ($)", secondary_y=True)
    
    return fig

@callback(
    Output('org-breakdown-chart', 'figure'),
    [Input('realtime-interval', 'n_intervals')],
    [State('org-hierarchy-store', 'data')]
)
def update_org_breakdown(n_intervals, org_data):
    if not org_data:
        return {}
    
    df = pd.DataFrame(org_data)
    
    # Create treemap for organizational cost breakdown
    fig = px.treemap(
        df,
        path=['name'],
        values='total_direct_costs',
        color='total_indirect_costs',
        color_continuous_scale='RdYlBu_r',
        title="Cost Distribution by Organization"
    )
    
    fig.update_layout(
        template="plotly_white",
        font_size=10
    )
    
    return fig
```


### Interactive Cost Calculator Page

The calculator page provides powerful cost modeling capabilities with real-time preview [^2_12][^2_13]:

```python
# pages/calculator.py
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from components.cost_calculator import AllocationMethod, CostRule

dash.register_page(__name__, path='/calculator', name='Cost Calculator')

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("ðŸ§® Advanced Cost Calculator", className="mb-4"),
                html.P("Design and test custom cost allocation scenarios", className="lead text-muted"),
                html.Hr()
            ])
        ]),
        
        # Calculator configuration panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("âš™ï¸ Configuration Panel"),
                    dbc.CardBody([
                        # Scenario selection
                        html.Label("Calculation Scenario:", className="fw-bold"),
                        dcc.Dropdown(
                            id='scenario-dropdown',
                            options=[
                                {'label': 'New Scenario', 'value': 'new'},
                                {'label': 'Budget Reallocation', 'value': 'budget'},
                                {'label': 'Department Restructure', 'value': 'restructure'},
                                {'label': 'Cost Center Consolidation', 'value': 'consolidation'},
                                {'label': 'Activity-Based Costing', 'value': 'abc'}
                            ],
                            value='new',
                            className="mb-3"
                        ),
                        
                        # Allocation method selection
                        html.Label("Allocation Method:", className="fw-bold"),
                        dcc.Dropdown(
                            id='allocation-method-dropdown',
                            options=[
                                {'label': 'Direct Allocation', 'value': 'direct'},
                                {'label': 'Step-Down Method', 'value': 'step_down'},
                                {'label': 'Reciprocal Method', 'value': 'reciprocal'},
                                {'label': 'Activity-Based Costing', 'value': 'activity_based'},
                                {'label': 'Usage-Based Allocation', 'value': 'usage_based'}
                            ],
                            value='direct',
                            className="mb-3"
                        ),
                        
                        # Cost pool inputs
                        html.Label("Cost Pool Amount ($):", className="fw-bold"),
                        dcc.Input(
                            id='cost-pool-amount',
                            type='number',
                            value=100000,
                            min=0,
                            step=1000,
                            className="form-control mb-3"
                        ),
                        
                        # Driver selection
                        html.Label("Allocation Driver:", className="fw-bold"),
                        dcc.Dropdown(
                            id='allocation-driver-dropdown',
                            options=[
                                {'label': 'Employee Headcount', 'value': 'headcount'},
                                {'label': 'Revenue', 'value': 'revenue'},
                                {'label': 'Square Footage', 'value': 'square_footage'},
                                {'label': 'CPU Hours', 'value': 'cpu_hours'},
                                {'label': 'Storage Usage', 'value': 'storage_usage'},
                                {'label': 'Transaction Volume', 'value': 'transaction_volume'}
                            ],
                            value='headcount',
                            className="mb-3"
                        ),
                        
                        # Advanced options
                        dbc.Accordion([
                            dbc.AccordionItem([
                                html.Label("Minimum Allocation ($):", className="fw-bold"),
                                dcc.Input(
                                    id='min-allocation',
                                    type='number',
                                    value=0,
                                    min=0,
                                    className="form-control mb-2"
                                ),
                                
                                html.Label("Maximum Allocation ($):", className="fw-bold"),
                                dcc.Input(
                                    id='max-allocation',
                                    type='number',
                                    value=None,
                                    min=0,
                                    className="form-control mb-2"
                                ),
                                
                                html.Label("Escalation Rate (%):", className="fw-bold"),
                                dcc.Input(
                                    id='escalation-rate',
                                    type='number',
                                    value=0,
                                    min=-100,
                                    max=100,
                                    step=0.1,
                                    className="form-control mb-2"
                                ),
                            ], title="Advanced Parameters"),
                        ], start_collapsed=True, className="mb-3"),
                        
                        # Action buttons
                        dbc.ButtonGroup([
                            dbc.Button("Calculate", id="calculate-btn", color="primary"),
                            dbc.Button("Reset", id="reset-btn", color="secondary", outline=True),
                            dbc.Button("Save Scenario", id="save-btn", color="success", outline=True),
                        ], className="d-grid gap-2")
                    ])
                ])
            ], width=4),
            
            # Results and visualization area
            dbc.Col([
                # Calculation results
                dbc.Card([
                    dbc.CardHeader("ðŸ“Š Calculation Results"),
                    dbc.CardBody([
                        dcc.Graph(id="calculation-results-chart", style={'height': '400px'})
                    ])
                ], className="mb-3"),
                
                # Detailed breakdown table
                dbc.Card([
                    dbc.CardHeader("ðŸ“‹ Detailed Allocation Breakdown"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='allocation-results-table',
                            columns=[
                                {'name': 'Organization', 'id': 'organization'},
                                {'name': 'Driver Value', 'id': 'driver_value', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                {'name': 'Allocation %', 'id': 'allocation_percentage', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                {'name': 'Allocated Cost', 'id': 'allocated_cost', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                {'name': 'Per Unit Cost', 'id': 'per_unit_cost', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                            ],
                            style_cell={'textAlign': 'center', 'fontSize': '12px'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 249, 250)'
                                }
                            ],
                            sort_action="native",
                            export_format="xlsx",
                            export_headers="display",
                            page_size=10
                        )
                    ])
                ])
            ], width=8)
        ], className="mb-4"),
        
        # Scenario comparison
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ðŸ”„ Scenario Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id="scenario-comparison-chart", style={'height': '300px'})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ðŸ“ˆ Sensitivity Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id="sensitivity-analysis-chart", style={'height': '300px'})
                    ])
                ])
            ], width=6)
        ]),
        
        # Store for calculation results
        dcc.Store(id='calculation-results-store'),
        dcc.Store(id='scenario-history-store', data=[])
    ], fluid=True)

@callback(
    [Output('calculation-results-store', 'data'),
     Output('calculation-results-chart', 'figure'),
     Output('allocation-results-table', 'data')],
    [Input('calculate-btn', 'n_clicks')],
    [State('scenario-dropdown', 'value'),
     State('allocation-method-dropdown', 'value'),
     State('cost-pool-amount', 'value'),
     State('allocation-driver-dropdown', 'value'),
     State('min-allocation', 'value'),
     State('max-allocation', 'value'),
     State('escalation-rate', 'value'),
     State('org-hierarchy-store', 'data')]
)
def calculate_allocation(n_clicks, scenario, method, cost_pool, driver, 
                        min_alloc, max_alloc, escalation, org_data):
    if n_clicks is None or not org_data:
        return {}, {}, []
    
    # Sample organizational data with drivers
    orgs = [
        {'name': 'Engineering', 'headcount': 150, 'revenue': 5000000, 'square_footage': 15000},
        {'name': 'Sales', 'headcount': 80, 'revenue': 12000000, 'square_footage': 8000},
        {'name': 'Marketing', 'headcount': 45, 'revenue': 2000000, 'square_footage': 5000},
        {'name': 'Operations', 'headcount': 60, 'revenue': 1000000, 'square_footage': 10000},
        {'name': 'Finance', 'headcount': 25, 'revenue': 500000, 'square_footage': 3000},
        {'name': 'HR', 'headcount': 15, 'revenue': 0, 'square_footage': 2000}
    ]
    
    df = pd.DataFrame(orgs)
    
    # Calculate allocation based on selected driver
    driver_column = driver
    total_driver_value = df[driver_column].sum()
    
    if total_driver_value == 0:
        return {}, {}, []
    
    # Calculate allocations
    df['driver_value'] = df[driver_column]
    df['allocation_percentage'] = df['driver_value'] / total_driver_value
    df['allocated_cost'] = df['allocation_percentage'] * cost_pool
    
    # Apply constraints
    if min_alloc:
        df['allocated_cost'] = df['allocated_cost'].clip(lower=min_alloc)
    if max_alloc:
        df['allocated_cost'] = df['allocated_cost'].clip(upper=max_alloc)
    
    # Apply escalation
    if escalation:
        df['allocated_cost'] = df['allocated_cost'] * (1 + escalation / 100)
    
    # Recalculate percentages after constraints
    total_allocated = df['allocated_cost'].sum()
    df['final_percentage'] = df['allocated_cost'] / total_allocated
    df['per_unit_cost'] = df['allocated_cost'] / df['driver_value']
    
    # Create visualization
    fig = px.bar(
        df, 
        x='name', 
        y='allocated_cost',
        color='allocation_percentage',
        color_continuous_scale='viridis',
        title=f"Cost Allocation Results - {method.replace('_', ' ').title()} Method",
        labels={'allocated_cost': 'Allocated Cost ($)', 'name': 'Organization'}
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    # Prepare table data
    table_data = df.rename(columns={
        'name': 'organization',
        'final_percentage': 'allocation_percentage'
    })[['organization', 'driver_value', 'allocation_percentage', 'allocated_cost', 'per_unit_cost']].to_dict('records')
    
    return df.to_dict('records'), fig, table_data

@callback(
    Output('sensitivity-analysis-chart', 'figure'),
    [Input('calculation-results-store', 'data')],
    [State('cost-pool-amount', 'value')]
)
def update_sensitivity_analysis(calc_results, base_cost_pool):
    if not calc_results:
        return {}
    
    df = pd.DataFrame(calc_results)
    
    # Generate sensitivity analysis for cost pool variations
    variations = np.arange(0.5, 2.1, 0.1)
    sensitivity_data = []
    
    for var in variations:
        varied_cost_pool = base_cost_pool * var
        for _, org in df.iterrows():
            sensitivity_data.append({
                'variation': var,
                'organization': org['name'],
                'allocated_cost': org['allocated_cost'] * var,
                'cost_pool_multiplier': var
            })
    
    sens_df = pd.DataFrame(sensitivity_data)
    
    fig = px.line(
        sens_df,
        x='variation',
        y='allocated_cost',
        color='organization',
        title="Sensitivity Analysis - Cost Pool Variations",
        labels={'variation': 'Cost Pool Multiplier', 'allocated_cost': 'Allocated Cost ($)'}
    )
    
    fig.update_layout(template="plotly_white")
    
    return fig
```

This comprehensive implementation provides a sophisticated multi-page dashboard with real-time Databricks integration, customizable cost calculation engines, and advanced visualization capabilities. The modular structure allows for easy extension and customization based on specific organizational requirements [^2_10][^2_3].

The dashboard supports multiple cost allocation methods, real-time data streaming, hierarchical organizational structures, and interactive cost modeling scenarios, making it a powerful tool for enterprise financial analysis and planning.
<span style="display:none">[^2_14][^2_15][^2_16][^2_17][^2_18][^2_19][^2_20][^2_21][^2_22][^2_23][^2_24][^2_25][^2_26][^2_27][^2_28][^2_29][^2_30][^2_31][^2_32][^2_33][^2_34][^2_35][^2_36][^2_37][^2_38][^2_39]</span>

<div align="center">â‚</div>

[^2_1]: https://dash.plotly.com/urls

[^2_2]: https://github.com/plotly/dash-dbx-sql

[^2_3]: https://community.plotly.com/t/structuring-a-large-dash-application-best-practices-to-follow/62739

[^2_4]: https://python.plainenglish.io/building-a-dockerised-structured-multipage-plotly-dash-app-3fb53c8703a0

[^2_5]: https://dash.plotly.com/databricks-integration/sql-connector

[^2_6]: https://dash.plotly.com/databricks-integration

[^2_7]: https://amnic.com/blogs/cloud-cost-allocation-methods

[^2_8]: https://fastercapital.com/content/Cost-Allocation-Algorithm--Cost-Allocation-Algorithms--A-Comprehensive-Guide.html

[^2_9]: https://www.youtube.com/watch?v=YU7bCEcsBK8

[^2_10]: https://www.marktechpost.com/2025/09/28/how-to-design-an-interactive-dash-and-plotly-dashboard-with-callback-mechanisms-for-local-and-online-deployment/

[^2_11]: https://www.tinybird.co/blog-posts/python-real-time-dashboard

[^2_12]: https://www.codearmo.com/python-tutorial/financial-option-pricing-dashboard-python-dash

[^2_13]: https://www.youtube.com/watch?v=Xgz0VWf9g2w

[^2_14]: https://plotly.com/examples/

[^2_15]: https://stackoverflow.com/questions/70241665/how-to-create-dashboard-with-multiple-pages-in-python

[^2_16]: https://dash.plotly.com/tutorial

[^2_17]: https://stackoverflow.com/questions/63589249/plotly-dash-display-real-time-data-in-smooth-animation

[^2_18]: https://www.planeks.net/website-development-cost-calculator/

[^2_19]: https://plotly.com/examples/dashboards/

[^2_20]: https://docs.databricks.com/aws/en/dev-tools/python-sql-connector

[^2_21]: https://pypi.org/project/openai-cost-calculator/

[^2_22]: https://community.databricks.com/t5/data-engineering/dash-in-databricks-notebook-directly/td-p/61830

[^2_23]: https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/

[^2_24]: https://estuary.dev/blog/connect-sql-server-to-databricks/

[^2_25]: https://www.scnsoft.com/software-development/python/calculator

[^2_26]: https://www.youtube.com/watch?v=GlRauKqI08Y

[^2_27]: https://towardsdatascience.com/create-a-dashboard-to-track-anything-with-plotly-and-dash-f9a5234d548b/

[^2_28]: https://stackoverflow.com/questions/57860270/use-dash-with-websockets

[^2_29]: https://www.qlik.com/us/dashboard-examples/financial-dashboards

[^2_30]: https://myadvantagecloud.cgi.com/micgr/fin/advantage/Advantage/Help/MainHelp/mergedProjects/Cost_Allocation/CA/Establish_the_Cost_Allocation_structure_Intro.htm

[^2_31]: https://community.plotly.com/t/is-it-possible-to-use-plotly-dash-with-real-time-data/90257

[^2_32]: https://www.youtube.com/watch?v=zHPaF8NoeuA

[^2_33]: https://github.com/AnnMarieW/dash-multi-page-app-demos

[^2_34]: https://www.highradius.com/resources/Blog/cost-allocation/

[^2_35]: https://community.plotly.com/t/live-update-from-a-database-file/79552

[^2_36]: https://www.costperform.com/cost-allocation-in-accounting/

[^2_37]: https://community.plotly.com/t/build-a-realtime-dash-app-with-websockets/48526

[^2_38]: https://kilmanndiagnostics.com/the-costs-of-organization-structure/

[^2_39]: https://stackoverflow.com/questions/65908128/streaming-live-websocket-stream-data-and-updating-plotly-graph-with-dash-within


---

# How can I configure and customize calculations and structures using this dashboard?

Building a configurable and customizable cost analysis dashboard requires multiple layers of configuration management. Here's a comprehensive guide on how to configure and customize calculations and structures in your Databricks dashboard.

## Configuration Architecture

The dashboard uses a multi-layered configuration system that supports JSON/YAML files, database-driven settings, and real-time updates [^3_1][^3_2]. The configuration is structured to handle organizational hierarchies, cost allocation rules, and user-specific customizations.

### Configuration File Structure

**Core Configuration Files:**

```python
# config/dashboard_config.yaml
dashboard:
  name: "Enterprise Cost Dashboard"
  refresh_interval: 30
  timezone: "UTC"
  theme: "bootstrap"

databricks:
  server_hostname: "${DATABRICKS_SERVER_HOSTNAME}"
  http_path: "${DATABRICKS_HTTP_PATH}"
  connection_pool_size: 10
  query_timeout: 300

organizational_structure:
  default_hierarchy_type: "functional"
  max_depth: 8
  cost_center_format: "CC-{:04d}"
  
cost_allocation:
  default_method: "step_down"
  minimum_allocation: 100.0
  escalation_rates:
    annual: 3.5
    quarterly: 0.875
  
calculation_rules:
  direct_costs:
    - type: "salary"
      allocation_driver: "headcount"
      frequency: "monthly"
    - type: "equipment"
      allocation_driver: "usage_hours"
      frequency: "daily"
  
  indirect_costs:
    - type: "facilities"
      allocation_driver: "square_footage"
      frequency: "monthly"
    - type: "utilities"
      allocation_driver: "headcount"
      frequency: "monthly"

user_interface:
  default_date_range: "90d"
  chart_themes:
    - "plotly_white"
    - "plotly_dark"
    - "seaborn"
  export_formats: ["xlsx", "csv", "pdf"]
```


### Configuration Management System

**Dynamic Configuration Loader:**

```python
# components/config_manager.py
import yaml
import json
import os
from typing import Dict, Any, Optional
import logging
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime
import jsonschema

class ConfigurationSchema(BaseModel):
    """Pydantic model for configuration validation"""
    
    class DatabaseConfig(BaseModel):
        server_hostname: str
        http_path: str
        connection_pool_size: int = Field(ge=1, le=50)
        query_timeout: int = Field(ge=60, le=3600)
    
    class OrganizationalConfig(BaseModel):
        default_hierarchy_type: str
        max_depth: int = Field(ge=2, le=10)
        cost_center_format: str
    
    class CostAllocationConfig(BaseModel):
        default_method: str
        minimum_allocation: float = Field(ge=0)
        escalation_rates: Dict[str, float]
    
    dashboard: Dict[str, Any]
    databricks: DatabaseConfig
    organizational_structure: OrganizationalConfig
    cost_allocation: CostAllocationConfig
    calculation_rules: Dict[str, Any]
    user_interface: Dict[str, Any]

class ConfigManager:
    def __init__(self, config_path: str = "config/dashboard_config.yaml"):
        self.config_path = config_path
        self.config_data: Optional[Dict] = None
        self.schema_path = "config/config_schema.json"
        self.custom_rules: Dict[str, Any] = {}
        self.organizational_hierarchies: Dict[str, Any] = {}
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load and validate configuration from file"""
        try:
            # Load YAML configuration
            with open(self.config_path, 'r') as file:
                raw_config = yaml.safe_load(file)
            
            # Expand environment variables
            expanded_config = self._expand_environment_variables(raw_config)
            
            # Validate configuration using Pydantic
            validated_config = ConfigurationSchema(**expanded_config)
            
            # Store validated configuration
            self.config_data = validated_config.dict()
            
            logging.info("Configuration loaded and validated successfully")
            return self.config_data
            
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except ValidationError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
    
    def _expand_environment_variables(self, config: Dict) -> Dict:
        """Recursively expand environment variables in configuration"""
        if isinstance(config, dict):
            return {key: self._expand_environment_variables(value) 
                   for key, value in config.items()}
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        elif isinstance(config, list):
            return [self._expand_environment_variables(item) for item in config]
        else:
            return config
    
    def get_cost_allocation_rules(self, rule_type: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve cost allocation rules with optional filtering"""
        if not self.config_data:
            self.load_configuration()
        
        rules = self.config_data.get('calculation_rules', {})
        
        if rule_type:
            return rules.get(rule_type, {})
        
        return rules
    
    def update_custom_rule(self, rule_id: str, rule_config: Dict[str, Any]) -> bool:
        """Update or create custom cost allocation rule"""
        try:
            # Validate rule configuration
            required_fields = ['name', 'allocation_method', 'driver_type']
            
            if not all(field in rule_config for field in required_fields):
                raise ValueError(f"Rule must contain: {required_fields}")
            
            # Store custom rule
            self.custom_rules[rule_id] = {
                **rule_config,
                'created_date': datetime.now().isoformat(),
                'modified_date': datetime.now().isoformat()
            }
            
            # Persist to database
            self._save_custom_rule_to_db(rule_id, rule_config)
            
            logging.info(f"Custom rule '{rule_id}' updated successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error updating custom rule: {e}")
            return False
    
    def get_organizational_hierarchy(self, hierarchy_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve organizational hierarchy configuration"""
        if hierarchy_id and hierarchy_id in self.organizational_hierarchies:
            return self.organizational_hierarchies[hierarchy_id]
        
        # Return default hierarchy from configuration
        return self.config_data.get('organizational_structure', {})
    
    def create_organizational_hierarchy(self, hierarchy_config: Dict[str, Any]) -> str:
        """Create new organizational hierarchy"""
        hierarchy_id = f"hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate hierarchy structure
        required_fields = ['name', 'structure_type', 'levels']
        
        if not all(field in hierarchy_config for field in required_fields):
            raise ValueError(f"Hierarchy must contain: {required_fields}")
        
        # Store hierarchy configuration
        self.organizational_hierarchies[hierarchy_id] = {
            **hierarchy_config,
            'id': hierarchy_id,
            'created_date': datetime.now().isoformat()
        }
        
        # Persist to database
        self._save_hierarchy_to_db(hierarchy_id, hierarchy_config)
        
        return hierarchy_id
```


## Settings Page Implementation

**Comprehensive Settings Interface:**

```python
# pages/settings.py
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import json
from components.config_manager import ConfigManager

dash.register_page(__name__, path='/settings', name='Settings & Configuration')

config_manager = ConfigManager()

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("âš™ï¸ Dashboard Settings & Configuration", className="mb-4"),
                html.P("Customize calculations, structures, and dashboard behavior", 
                       className="lead text-muted"),
                html.Hr()
            ])
        ]),
        
        # Configuration Tabs
        dbc.Tabs([
            # Organizational Structure Tab
            dbc.Tab(label="ðŸ¢ Organizational Structure", tab_id="org-structure", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Create/Edit Organizational Hierarchy"),
                                dbc.CardBody([
                                    html.Label("Hierarchy Name:", className="fw-bold"),
                                    dcc.Input(
                                        id='hierarchy-name-input',
                                        type='text',
                                        placeholder='e.g., Regional Structure',
                                        className="form-control mb-3"
                                    ),
                                    
                                    html.Label("Structure Type:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='structure-type-dropdown',
                                        options=[
                                            {'label': 'Functional', 'value': 'functional'},
                                            {'label': 'Divisional', 'value': 'divisional'},
                                            {'label': 'Matrix', 'value': 'matrix'},
                                            {'label': 'Geographic', 'value': 'geographic'},
                                            {'label': 'Product-based', 'value': 'product_based'}
                                        ],
                                        value='functional',
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Maximum Hierarchy Depth:", className="fw-bold"),
                                    dcc.Input(
                                        id='max-depth-input',
                                        type='number',
                                        value=5,
                                        min=2,
                                        max=10,
                                        className="form-control mb-3"
                                    ),
                                    
                                    html.Label("Cost Center Naming Convention:", className="fw-bold"),
                                    dcc.Input(
                                        id='cost-center-format-input',
                                        type='text',
                                        value='CC-{:04d}',
                                        placeholder='e.g., CC-{:04d}',
                                        className="form-control mb-3"
                                    ),
                                    
                                    dbc.Button("Save Hierarchy", id="save-hierarchy-btn", 
                                              color="primary", className="me-2"),
                                    dbc.Button("Load Template", id="load-template-btn", 
                                              color="secondary", outline=True),
                                ])
                            ])
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Hierarchy Visualization"),
                                dbc.CardBody([
                                    dcc.Graph(id="hierarchy-preview-chart", 
                                             style={'height': '400px'})
                                ])
                            ])
                        ], width=6)
                    ], className="mb-4"),
                    
                    # Organizational Levels Configuration
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Configure Hierarchy Levels"),
                                dbc.CardBody([
                                    html.Div(id="hierarchy-levels-container"),
                                    dbc.Button("Add Level", id="add-level-btn", 
                                              color="success", size="sm", className="mt-2")
                                ])
                            ])
                        ])
                    ])
                ], className="p-3")
            ]),
            
            # Cost Allocation Rules Tab
            dbc.Tab(label="ðŸ’° Cost Allocation Rules", tab_id="cost-rules", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Allocation Method Configuration"),
                                dbc.CardBody([
                                    html.Label("Default Allocation Method:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='default-allocation-method',
                                        options=[
                                            {'label': 'Direct Allocation', 'value': 'direct'},
                                            {'label': 'Step-Down Method', 'value': 'step_down'},
                                            {'label': 'Reciprocal Method', 'value': 'reciprocal'},
                                            {'label': 'Activity-Based Costing', 'value': 'activity_based'},
                                            {'label': 'Usage-Based Allocation', 'value': 'usage_based'}
                                        ],
                                        value='step_down',
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Minimum Allocation Amount ($):", className="fw-bold"),
                                    dcc.Input(
                                        id='min-allocation-global',
                                        type='number',
                                        value=100,
                                        min=0,
                                        step=10,
                                        className="form-control mb-3"
                                    ),
                                    
                                    html.Label("Annual Escalation Rate (%):", className="fw-bold"),
                                    dcc.Input(
                                        id='annual-escalation-rate',
                                        type='number',
                                        value=3.5,
                                        min=0,
                                        max=20,
                                        step=0.1,
                                        className="form-control mb-3"
                                    ),
                                ])
                            ])
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Allocation Drivers Configuration"),
                                dbc.CardBody([
                                    html.Div(id="allocation-drivers-container"),
                                    dbc.Button("Add Driver", id="add-driver-btn", 
                                              color="success", size="sm", className="mt-2")
                                ])
                            ])
                        ], width=6)
                    ], className="mb-4"),
                    
                    # Custom Rules Table
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("Custom Allocation Rules", className="mb-0"),
                                    dbc.Button("Create New Rule", id="create-rule-btn", 
                                              color="primary", size="sm")
                                ], className="d-flex justify-content-between align-items-center"),
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        id='custom-rules-table',
                                        columns=[
                                            {'name': 'Rule Name', 'id': 'name', 'editable': True},
                                            {'name': 'Method', 'id': 'allocation_method', 
                                             'presentation': 'dropdown',
                                             'editable': True},
                                            {'name': 'Driver', 'id': 'driver_type', 'editable': True},
                                            {'name': 'Min Amount', 'id': 'min_allocation', 
                                             'type': 'numeric', 'editable': True},
                                            {'name': 'Max Amount', 'id': 'max_allocation', 
                                             'type': 'numeric', 'editable': True},
                                            {'name': 'Status', 'id': 'status'},
                                            {'name': 'Actions', 'id': 'actions', 'presentation': 'markdown'}
                                        ],
                                        data=[],
                                        editable=True,
                                        row_deletable=True,
                                        style_cell={'textAlign': 'left', 'fontSize': '12px'},
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 
                                                     'fontWeight': 'bold'},
                                        dropdown={
                                            'allocation_method': {
                                                'options': [
                                                    {'label': 'Direct', 'value': 'direct'},
                                                    {'label': 'Step-Down', 'value': 'step_down'},
                                                    {'label': 'Reciprocal', 'value': 'reciprocal'},
                                                    {'label': 'Activity-Based', 'value': 'activity_based'},
                                                    {'label': 'Usage-Based', 'value': 'usage_based'}
                                                ]
                                            }
                                        }
                                    )
                                ])
                            ])
                        ])
                    ])
                ], className="p-3")
            ]),
            
            # Calculation Settings Tab
            dbc.Tab(label="ðŸ§® Calculation Settings", tab_id="calc-settings", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Direct Cost Categories"),
                                dbc.CardBody([
                                    html.Div(id="direct-cost-categories"),
                                    dbc.Button("Add Category", id="add-direct-category-btn", 
                                              color="success", size="sm", className="mt-2")
                                ])
                            ])
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Indirect Cost Categories"),
                                dbc.CardBody([
                                    html.Div(id="indirect-cost-categories"),
                                    dbc.Button("Add Category", id="add-indirect-category-btn", 
                                              color="success", size="sm", className="mt-2")
                                ])
                            ])
                        ], width=6)
                    ], className="mb-4"),
                    
                    # Formula Builder
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Custom Formula Builder"),
                                dbc.CardBody([
                                    html.Label("Formula Name:", className="fw-bold"),
                                    dcc.Input(
                                        id='formula-name-input',
                                        type='text',
                                        placeholder='e.g., Weighted Average Allocation',
                                        className="form-control mb-3"
                                    ),
                                    
                                    html.Label("Formula Expression:", className="fw-bold"),
                                    dcc.Textarea(
                                        id='formula-expression-input',
                                        placeholder='e.g., (direct_costs + indirect_costs * 0.8) / headcount',
                                        style={'height': '100px'},
                                        className="form-control mb-3"
                                    ),
                                    
                                    html.Label("Variables:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='formula-variables-dropdown',
                                        options=[
                                            {'label': 'Direct Costs', 'value': 'direct_costs'},
                                            {'label': 'Indirect Costs', 'value': 'indirect_costs'},
                                            {'label': 'Headcount', 'value': 'headcount'},
                                            {'label': 'Revenue', 'value': 'revenue'},
                                            {'label': 'Square Footage', 'value': 'square_footage'},
                                            {'label': 'CPU Hours', 'value': 'cpu_hours'}
                                        ],
                                        multi=True,
                                        className="mb-3"
                                    ),
                                    
                                    dbc.Button("Test Formula", id="test-formula-btn", 
                                              color="info", className="me-2"),
                                    dbc.Button("Save Formula", id="save-formula-btn", 
                                              color="primary")
                                ])
                            ])
                        ])
                    ])
                ], className="p-3")
            ]),
            
            # Dashboard Settings Tab
            dbc.Tab(label="ðŸŽ›ï¸ Dashboard Settings", tab_id="dashboard-settings", children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Display Settings"),
                                dbc.CardBody([
                                    html.Label("Default Date Range:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='default-date-range',
                                        options=[
                                            {'label': '7 Days', 'value': '7d'},
                                            {'label': '30 Days', 'value': '30d'},
                                            {'label': '90 Days', 'value': '90d'},
                                            {'label': '1 Year', 'value': '1y'},
                                            {'label': 'Year to Date', 'value': 'ytd'}
                                        ],
                                        value='90d',
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Chart Theme:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='chart-theme',
                                        options=[
                                            {'label': 'Light', 'value': 'plotly_white'},
                                            {'label': 'Dark', 'value': 'plotly_dark'},
                                            {'label': 'Seaborn', 'value': 'seaborn'},
                                            {'label': 'Ggplot2', 'value': 'ggplot2'}
                                        ],
                                        value='plotly_white',
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Auto-Refresh Interval (seconds):", className="fw-bold"),
                                    dcc.Slider(
                                        id='refresh-interval-slider',
                                        min=10,
                                        max=300,
                                        step=10,
                                        value=30,
                                        marks={
                                            10: '10s',
                                            30: '30s',
                                            60: '1m',
                                            120: '2m',
                                            300: '5m'
                                        },
                                        className="mb-3"
                                    ),
                                ])
                            ])
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Export Settings"),
                                dbc.CardBody([
                                    html.Label("Default Export Format:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='export-format',
                                        options=[
                                            {'label': 'Excel (.xlsx)', 'value': 'xlsx'},
                                            {'label': 'CSV (.csv)', 'value': 'csv'},
                                            {'label': 'PDF Report', 'value': 'pdf'},
                                            {'label': 'JSON Data', 'value': 'json'}
                                        ],
                                        value='xlsx',
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Include Charts in Export:", className="fw-bold"),
                                    dbc.Switch(
                                        id='include-charts-switch',
                                        label="Yes",
                                        value=True,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Data Precision (Decimal Places):", className="fw-bold"),
                                    dcc.Input(
                                        id='data-precision-input',
                                        type='number',
                                        value=2,
                                        min=0,
                                        max=6,
                                        className="form-control mb-3"
                                    ),
                                ])
                            ])
                        ], width=6)
                    ])
                ], className="p-3")
            ])
        ], id="settings-tabs", active_tab="org-structure", className="mb-4"),
        
        # Action Buttons
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("ðŸ’¾ Save All Settings", id="save-all-settings-btn", 
                              color="primary", size="lg"),
                    dbc.Button("ðŸ”„ Reset to Defaults", id="reset-defaults-btn", 
                              color="secondary", outline=True, size="lg"),
                    dbc.Button("ðŸ“ Export Configuration", id="export-config-btn", 
                              color="info", outline=True, size="lg"),
                    dbc.Button("ðŸ“‚ Import Configuration", id="import-config-btn", 
                              color="success", outline=True, size="lg"),
                ])
            ], className="text-center")
        ]),
        
        # Hidden file upload for configuration import
        dcc.Upload(
            id='config-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Configuration File')
            ]),
            style={
                'display': 'none'
            }
        ),
        
        # Store components for configuration state
        dcc.Store(id='current-config-store'),
        dcc.Store(id='hierarchy-levels-store', data=[]),
        dcc.Store(id='custom-rules-store', data=[]),
    ], fluid=True)

# Dynamic configuration callbacks
@callback(
    [Output('hierarchy-levels-container', 'children'),
     Output('hierarchy-levels-store', 'data')],
    [Input('add-level-btn', 'n_clicks'),
     Input('structure-type-dropdown', 'value')],
    [State('hierarchy-levels-store', 'data'),
     State('max-depth-input', 'value')]
)
def manage_hierarchy_levels(add_clicks, structure_type, current_levels, max_depth):
    """Dynamically manage organizational hierarchy levels"""
    
    if not current_levels:
        # Initialize with default levels based on structure type
        if structure_type == 'functional':
            current_levels = [
                {'level': 1, 'name': 'Company', 'code_prefix': 'CO'},
                {'level': 2, 'name': 'Division', 'code_prefix': 'DIV'},
                {'level': 3, 'name': 'Department', 'code_prefix': 'DEPT'}
            ]
        elif structure_type == 'geographic':
            current_levels = [
                {'level': 1, 'name': 'Global', 'code_prefix': 'GL'},
                {'level': 2, 'name': 'Region', 'code_prefix': 'REG'},
                {'level': 3, 'name': 'Country', 'code_prefix': 'CTY'}
            ]
        else:
            current_levels = [
                {'level': 1, 'name': 'Organization', 'code_prefix': 'ORG'},
                {'level': 2, 'name': 'Unit', 'code_prefix': 'UNIT'}
            ]
    
    # Add new level if button clicked
    ctx = dash.callback_context
    if ctx.triggered and 'add-level-btn' in ctx.triggered[^3_0]['prop_id']:
        if len(current_levels) < max_depth:
            new_level = {
                'level': len(current_levels) + 1,
                'name': f'Level {len(current_levels) + 1}',
                'code_prefix': f'L{len(current_levels) + 1}'
            }
            current_levels.append(new_level)
    
    # Generate UI components for each level
    level_components = []
    for i, level in enumerate(current_levels):
        level_component = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label(f"Level {level['level']} Name:", className="fw-bold"),
                        dcc.Input(
                            id=f"level-{i}-name",
                            value=level['name'],
                            className="form-control"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Code Prefix:", className="fw-bold"),
                        dcc.Input(
                            id=f"level-{i}-prefix",
                            value=level['code_prefix'],
                            className="form-control"
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Cost Allocation:", className="fw-bold"),
                        dcc.Dropdown(
                            id=f"level-{i}-allocation",
                            options=[
                                {'label': 'Direct', 'value': 'direct'},
                                {'label': 'Allocated', 'value': 'allocated'},
                                {'label': 'Both', 'value': 'both'}
                            ],
                            value='both'
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Button("Ã—", id=f"remove-level-{i}-btn", 
                                  color="danger", size="sm", 
                                  className="mt-4")
                    ], width=2)
                ])
            ])
        ], className="mb-2")
        
        level_components.append(level_component)
    
    return level_components, current_levels

@callback(
    Output('custom-rules-table', 'data'),
    [Input('create-rule-btn', 'n_clicks')],
    [State('custom-rules-store', 'data')]
)
def manage_custom_rules(create_clicks, current_rules):
    """Manage custom allocation rules dynamically"""
    
    ctx = dash.callback_context
    if ctx.triggered and 'create-rule-btn' in ctx.triggered[^3_0]['prop_id']:
        new_rule = {
            'id': f"rule_{len(current_rules) + 1}",
            'name': f'Custom Rule {len(current_rules) + 1}',
            'allocation_method': 'direct',
            'driver_type': 'headcount',
            'min_allocation': 0,
            'max_allocation': None,
            'status': 'Active',
            'actions': '[Edit](#{}) | [Delete](#{})'
        }
        current_rules.append(new_rule)
    
    return current_rules

@callback(
    Output('current-config-store', 'data'),
    [Input('save-all-settings-btn', 'n_clicks')],
    [State('hierarchy-name-input', 'value'),
     State('structure-type-dropdown', 'value'),
     State('max-depth-input', 'value'),
     State('default-allocation-method', 'value'),
     State('min-allocation-global', 'value'),
     State('annual-escalation-rate', 'value'),
     State('hierarchy-levels-store', 'data'),
     State('custom-rules-store', 'data')]
)
def save_configuration(save_clicks, hierarchy_name, structure_type, max_depth,
                      allocation_method, min_allocation, escalation_rate,
                      hierarchy_levels, custom_rules):
    """Save all configuration settings"""
    
    if save_clicks is None:
        return {}
    
    # Build configuration dictionary
    config = {
        'organizational_structure': {
            'hierarchy_name': hierarchy_name,
            'structure_type': structure_type,
            'max_depth': max_depth,
            'levels': hierarchy_levels
        },
        'cost_allocation': {
            'default_method': allocation_method,
            'minimum_allocation': min_allocation,
            'escalation_rates': {'annual': escalation_rate}
        },
        'custom_rules': custom_rules,
        'last_updated': datetime.now().isoformat()
    }
    
    # Save to configuration manager
    config_manager.update_configuration(config)
    
    return config
```


## Dynamic Data-Driven Components

**Database-Driven Dropdown Options:**

```python
# Real-time dropdown options from Databricks
@callback(
    Output('org-selector-dropdown', 'options'),
    [Input('realtime-interval', 'n_intervals')]
)
def update_organization_options(n_intervals):
    """Dynamically load organization options from database"""
    
    query = """
    SELECT DISTINCT 
        id as value, 
        name as label,
        level,
        parent_id
    FROM organizations 
    WHERE active = true
    ORDER BY level, name
    """
    
    try:
        results = db_client.execute_query(query)
        options = [{'label': row[^3_1], 'value': row[^3_0]} for row in results]
        return options
    except Exception as e:
        logging.error(f"Error loading organization options: {e}")
        return [{'label': 'Error loading options', 'value': None}]

@callback(
    Output('cost-driver-dropdown', 'options'),
    [Input('allocation-method-dropdown', 'value')]
)
def update_driver_options(allocation_method):
    """Update available drivers based on selected allocation method"""
    
    driver_mappings = {
        'direct': [
            {'label': 'Employee Headcount', 'value': 'headcount'},
            {'label': 'Revenue', 'value': 'revenue'}
        ],
        'step_down': [
            {'label': 'Employee Headcount', 'value': 'headcount'},
            {'label': 'Square Footage', 'value': 'square_footage'},
            {'label': 'Revenue', 'value': 'revenue'}
        ],
        'activity_based': [
            {'label': 'Transaction Volume', 'value': 'transaction_volume'},
            {'label': 'CPU Hours', 'value': 'cpu_hours'},
            {'label': 'Storage Usage', 'value': 'storage_usage'}
        ],
        'usage_based': [
            {'label': 'CPU Hours', 'value': 'cpu_hours'},
            {'label': 'Storage Usage', 'value': 'storage_usage'},
            {'label': 'Network Bandwidth', 'value': 'network_bandwidth'}
        ]
    }
    
    return driver_mappings.get(allocation_method, [])
```


## Configuration Validation and Persistence

**Configuration validation ensures data integrity and prevents errors**:

```python
# Configuration validation schema
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "organizational_structure": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "minimum": 2, "maximum": 10},
                "levels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "integer"},
                            "name": {"type": "string"},
                            "code_prefix": {"type": "string"}
                        },
                        "required": ["level", "name", "code_prefix"]
                    }
                }
            }
        },
        "cost_allocation": {
            "type": "object",
            "properties": {
                "minimum_allocation": {"type": "number", "minimum": 0},
                "escalation_rates": {
                    "type": "object",
                    "properties": {
                        "annual": {"type": "number", "minimum": 0, "maximum": 50}
                    }
                }
            }
        }
    }
}

def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate configuration against schema"""
    try:
        jsonschema.validate(config, CONFIG_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
```

This comprehensive configuration system provides [^3_2][^3_3][^3_4]:

### **Key Customization Features**

1. **Hierarchical Structure Management** - Create unlimited organizational levels with custom naming conventions [^3_5][^3_6]
2. **Dynamic Cost Allocation Rules** - Configure multiple allocation methods with custom formulas [^3_7][^3_8]
3. **Real-time Configuration Updates** - Changes reflect immediately across the dashboard [^3_4][^3_9]
4. **Data-Driven Components** - Dropdown options populate dynamically from Databricks [^3_4][^3_10]
5. **Configuration Validation** - Ensures data integrity with schema-based validation [^3_11][^3_3]
6. **Import/Export Functionality** - Share configurations across environments [^3_1][^3_2]

The system supports complete customization of calculations, organizational structures, and dashboard behavior while maintaining data consistency and user-friendly interfaces for configuration management.
<span style="display:none">[^3_12][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_19][^3_20][^3_21][^3_22][^3_23][^3_24][^3_25][^3_26][^3_27][^3_28][^3_29][^3_30][^3_31][^3_32][^3_33][^3_34][^3_35][^3_36][^3_37][^3_38][^3_39][^3_40]</span>

<div align="center">â‚</div>

[^3_1]: https://configu.com/blog/working-with-python-configuration-files-tutorial-best-practices/

[^3_2]: https://www.lambdatest.com/blog/python-configuration-file/

[^3_3]: https://betterstack.com/community/guides/scaling-python/yaml-files-in-python/

[^3_4]: https://community.plotly.com/t/dynamic-options-for-drop-down/7550

[^3_5]: https://www.accountingtools.com/articles/hierarchical-organizational-structure

[^3_6]: https://www.functionly.com/orginometry/hierarchy/hierarchical-structure

[^3_7]: https://www.youtube.com/watch?v=FWfjprUBtvk

[^3_8]: https://docs.oracle.com/en/cloud/saas/financials/24c/faiac/example-of-creating-an-allocation-rule-and-generating.html

[^3_9]: https://dash.plotly.com/dash-core-components/dropdown

[^3_10]: https://stackoverflow.com/questions/69884482/dynamically-updating-dropdown-options-with-2-inputs-in-callback-dash

[^3_11]: https://www.rapids.science/1.10/developers/validation-schema-config/

[^3_12]: https://stackoverflow.com/questions/49643793/what-is-the-best-method-for-setting-up-a-config-file-in-python

[^3_13]: https://www.geeksforgeeks.org/how-to-write-a-configuration-file-in-python/

[^3_14]: https://www.reddit.com/r/Python/comments/w1utza/how_are_you_all_handling_config_files/

[^3_15]: https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/

[^3_16]: https://pypi.org/project/python-configuration-management/

[^3_17]: https://help.sap.com/docs/REAL_ESTATE_DEVELOPMENT_LIFECYCLE_MANAGEMENT_S4HANA/10d72174fe994f0e8e82a8ac7c4eb041/008ba961a8894db087235a081d712f53.html?locale=en-US\&state=PRODUCTION\&version=1812

[^3_18]: https://www.ebsco.com/research-starters/business-and-management/hierarchical-organizational-structure

[^3_19]: https://www.browserstack.com/guide/read-config-files-in-python-and-selenium

[^3_20]: https://learn.microsoft.com/en-us/dynamics365/finance/general-ledger/ledger-allocation-rules

[^3_21]: https://www.canva.com/online-whiteboard/organizational-charts/

[^3_22]: https://www.tothenew.com/blog/integrating-python-json-logger-with-grafana-and-loki-for-enhanced-logging/

[^3_23]: https://community.sap.com/t5/financial-management-blog-posts-by-sap/profitability-and-cost-allocation-leveraging-dynamic-rule-maintenance/ba-p/13744173

[^3_24]: https://whatfix.com/blog/organizational-structure/

[^3_25]: https://learn.microsoft.com/en-us/dynamics365/supply-chain/cost-management/methodology-total-cost-allocation

[^3_26]: https://www.educative.io/answers/how-to-create-a-dropdown-component-using-dash-in-python

[^3_27]: https://www.nobledesktop.com/learn/python/interactive-menus-dash

[^3_28]: https://www.forestadmin.com/blog/flask-tastic-admin-panel-a-step-by-step-guide-to-building-your-own-2/

[^3_29]: https://github.com/plotly/dash/issues/2487

[^3_30]: https://flask-admin.readthedocs.io/en/v1.0.8/quickstart/

[^3_31]: https://www.youtube.com/watch?v=UYH_dNSX1DM

[^3_32]: https://flask-admin.readthedocs.io

[^3_33]: https://betterprogramming.pub/validating-yaml-configs-made-easy-with-pydantic-594522612db5

[^3_34]: https://flask-admin.readthedocs.io/en/stable/introduction/

[^3_35]: https://stackoverflow.com/questions/3262569/validating-a-yaml-document-in-python

[^3_36]: https://www.youtube.com/watch?v=8BB3UK_pQy8

[^3_37]: https://www.andrewvillazon.com/validate-yaml-python-schema/

[^3_38]: https://github.com/jonalxh/Flask-Admin-Dashboard

[^3_39]: https://www.codethink.co.uk/articles/2021/yaml-schemas/

[^3_40]: http://docs.peewee-orm.com/projects/flask-peewee/en/latest/admin.html


---

# Complete Installation \& Setup Guide for Enterprise Cost Dashboard

This comprehensive guide will walk you through installing, configuring, and connecting your data to the Enterprise Cost Dashboard with Databricks integration.

## Prerequisites and System Requirements

### System Requirements

**Minimum Requirements:**

- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Network access to Databricks workspace
- Modern web browser (Chrome, Firefox, Safari, Edge)

**Databricks Requirements:**

- Databricks workspace with SQL Warehouse or All-Purpose Cluster
- Personal Access Token or OAuth authentication
- Database permissions for schema creation and data access


### Required Software

Install the following software before proceeding [^4_1][^4_2]:

```bash
# Check Python version (must be 3.8+)
python --version
# or
python3 --version

# Install Git (if not already installed)
# On Ubuntu/Debian
sudo apt update && sudo apt install git

# On macOS
brew install git

# On Windows - download from https://git-scm.com/
```


## Step 1: Project Setup and Environment Creation

### Clone or Create Project Directory

```bash
# Create project directory
mkdir enterprise-cost-dashboard
cd enterprise-cost-dashboard

# Initialize Git repository
git init
```


### Create Virtual Environment

Create an isolated Python environment to prevent dependency conflicts [^4_1][^4_3]:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Verify activation (should show .venv path)
which python
```


### Create Project Structure

Set up the recommended project structure:

```bash
# Create directory structure
mkdir -p {pages,components,assets,config,data,logs,tests}
mkdir -p docker/production

# Create essential files
touch app.py requirements.txt .env.example .env .gitignore
touch config/dashboard_config.yaml config/config_schema.json
```


## Step 2: Install Dependencies

### Create Requirements File

Create `requirements.txt` with all necessary dependencies [^4_4][^4_3]:

```txt
# Core Dashboard Framework
dash==2.17.1
dash-bootstrap-components==1.5.0
plotly==5.17.0
pandas==2.1.3
numpy==1.25.2

# Databricks Integration
databricks-sql-connector[pyarrow]==3.0.3
pyarrow==14.0.1
sqlalchemy==2.0.23

# Configuration Management
pyyaml==6.0.1
python-dotenv==1.0.0
pydantic==2.5.0
jsonschema==4.20.0

# Real-time Updates
redis==5.0.1
celery==5.3.4

# Logging and Monitoring
structlog==23.2.0
python-json-logger==2.0.7

# Production Dependencies
gunicorn==21.2.0
gevent==23.9.1

# Development Dependencies (Optional)
pytest==7.4.3
black==23.11.0
flake8==6.1.0
pre-commit==3.6.0
jupyter==1.0.0

# Security
cryptography==41.0.8
werkzeug==3.0.1
```


### Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```


## Step 3: Databricks Connection Configuration

### Gather Databricks Connection Information

1. **Access Your Databricks Workspace:**
    - Log into your Databricks workspace
    - Go to **SQL Warehouses** or **Compute** (for All-Purpose Clusters)
2. **Get Connection Details** [^4_5][^4_6]:
    - **Server Hostname**: Found in Connection Details tab
    - **HTTP Path**: Found in Connection Details tab (e.g., `/sql/1.0/endpoints/1234567890abcdef`)
    - **Access Token**: Generate from User Settings â†’ Access Tokens

### Create Environment Configuration

Create `.env` file with your Databricks credentials [^4_7][^4_8]:

```bash
# Copy example file
cp .env.example .env
```

**`.env` file contents:**

```bash
# Databricks Connection
DATABRICKS_SERVER_HOSTNAME=your-workspace.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/endpoints/your-endpoint-id
DATABRICKS_TOKEN=your-personal-access-token

# Application Settings
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-change-in-production

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
REFRESH_INTERVAL=30

# Cache Configuration (Optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TIMEOUT=300

# Database Schema
DEFAULT_CATALOG=main
DEFAULT_SCHEMA=cost_analysis
```

**`.env.example` file (for version control):**

```bash
# Databricks Connection
DATABRICKS_SERVER_HOSTNAME=your-workspace.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/endpoints/your-endpoint-id
DATABRICKS_TOKEN=your-personal-access-token

# Application Settings
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
SECRET_KEY=change-me-in-production

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
REFRESH_INTERVAL=30
```


### Test Databricks Connection

Create a connection test script:

```python
# test_connection.py
import os
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def test_databricks_connection():
    try:
        connection = sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_timestamp() as test_time")
            result = cursor.fetchone()
            print(f"âœ… Connection successful! Current time: {result[^4_0]}")
            
        connection.close()
        return True
        
    except Exception as e:
        print(f"âŒ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_databricks_connection()
```

Run the test:

```bash
python test_connection.py
```


## Step 4: Database Schema Setup

### Create Database Schema

Create the necessary database schema in Databricks [^4_9][^4_10]:

```sql
-- Create schema and tables in Databricks SQL Editor or Notebook

-- Create main schema
CREATE SCHEMA IF NOT EXISTS cost_analysis
COMMENT 'Enterprise Cost Analysis Dashboard Schema';

-- Use the schema
USE cost_analysis;

-- Organizations table
CREATE TABLE IF NOT EXISTS organizations (
    id STRING PRIMARY KEY,
    name STRING NOT NULL,
    parent_id STRING,
    level INTEGER NOT NULL,
    cost_center_code STRING,
    structure_type STRING DEFAULT 'functional',
    direct_cost_rate DECIMAL(10,4) DEFAULT 0.0,
    indirect_cost_rate DECIMAL(10,4) DEFAULT 0.0,
    active BOOLEAN DEFAULT true,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) COMMENT 'Organizational hierarchy structure';

-- Cost allocations table
CREATE TABLE IF NOT EXISTS cost_allocations (
    id STRING PRIMARY KEY,
    org_id STRING NOT NULL,
    transaction_date DATE NOT NULL,
    direct_costs DECIMAL(15,2) DEFAULT 0.0,
    indirect_costs DECIMAL(15,2) DEFAULT 0.0,
    allocated_costs DECIMAL(15,2) DEFAULT 0.0,
    cost_pool_id STRING,
    allocation_method STRING,
    transaction_id STRING,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) COMMENT 'Cost allocation transactions';

-- Cost categories table
CREATE TABLE IF NOT EXISTS cost_categories (
    id STRING PRIMARY KEY,
    name STRING NOT NULL,
    parent_id STRING,
    category_type STRING NOT NULL, -- 'direct', 'indirect'
    allocation_driver STRING, -- 'headcount', 'revenue', etc.
    active BOOLEAN DEFAULT true,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) COMMENT 'Cost category hierarchy';

-- Allocation drivers table
CREATE TABLE IF NOT EXISTS allocation_drivers (
    id STRING PRIMARY KEY,
    org_id STRING NOT NULL,
    driver_type STRING NOT NULL,
    driver_value DECIMAL(15,4) NOT NULL,
    measurement_date DATE NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) COMMENT 'Allocation driver values by organization';

-- Cost rules table
CREATE TABLE IF NOT EXISTS cost_rules (
    id STRING PRIMARY KEY,
    rule_name STRING NOT NULL,
    allocation_method STRING NOT NULL,
    driver_type STRING NOT NULL,
    source_cost_center STRING,
    target_cost_centers STRING, -- JSON array
    allocation_percentage DECIMAL(5,4),
    minimum_allocation DECIMAL(15,2),
    maximum_allocation DECIMAL(15,2),
    escalation_rate DECIMAL(5,4) DEFAULT 0.0,
    effective_date DATE NOT NULL,
    expiry_date DATE,
    active BOOLEAN DEFAULT true,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) COMMENT 'Custom cost allocation rules';

-- Add foreign key constraints
ALTER TABLE organizations ADD CONSTRAINT fk_org_parent 
    FOREIGN KEY (parent_id) REFERENCES organizations(id);

ALTER TABLE cost_allocations ADD CONSTRAINT fk_allocation_org 
    FOREIGN KEY (org_id) REFERENCES organizations(id);

ALTER TABLE allocation_drivers ADD CONSTRAINT fk_driver_org 
    FOREIGN KEY (org_id) REFERENCES organizations(id);
```


### Load Sample Data

Create sample data for testing:

```sql
-- Insert sample organizations
INSERT INTO organizations (id, name, parent_id, level, cost_center_code) VALUES
('org_001', 'Enterprise Corp', NULL, 1, 'CC-0001'),
('org_002', 'Engineering Division', 'org_001', 2, 'CC-0002'),
('org_003', 'Sales Division', 'org_001', 2, 'CC-0003'),
('org_004', 'Software Engineering', 'org_002', 3, 'CC-0004'),
('org_005', 'Data Engineering', 'org_002', 3, 'CC-0005'),
('org_006', 'Enterprise Sales', 'org_003', 3, 'CC-0006'),
('org_007', 'Inside Sales', 'org_003', 3, 'CC-0007');

-- Insert sample cost categories
INSERT INTO cost_categories (id, name, parent_id, category_type, allocation_driver) VALUES
('cat_001', 'Direct Costs', NULL, 'direct', 'direct'),
('cat_002', 'Personnel', 'cat_001', 'direct', 'headcount'),
('cat_003', 'Equipment', 'cat_001', 'direct', 'usage_hours'),
('cat_004', 'Indirect Costs', NULL, 'indirect', 'allocated'),
('cat_005', 'Facilities', 'cat_004', 'indirect', 'square_footage'),
('cat_006', 'Utilities', 'cat_004', 'indirect', 'headcount');

-- Insert sample allocation drivers
INSERT INTO allocation_drivers (id, org_id, driver_type, driver_value, measurement_date) VALUES
('drv_001', 'org_002', 'headcount', 150.0, CURRENT_DATE()),
('drv_002', 'org_002', 'revenue', 5000000.0, CURRENT_DATE()),
('drv_003', 'org_002', 'square_footage', 15000.0, CURRENT_DATE()),
('drv_004', 'org_003', 'headcount', 80.0, CURRENT_DATE()),
('drv_005', 'org_003', 'revenue', 12000000.0, CURRENT_DATE()),
('drv_006', 'org_003', 'square_footage', 8000.0, CURRENT_DATE());

-- Insert sample cost allocations
INSERT INTO cost_allocations (id, org_id, transaction_date, direct_costs, indirect_costs, allocated_costs) VALUES
('alloc_001', 'org_002', CURRENT_DATE(), 500000.0, 150000.0, 650000.0),
('alloc_002', 'org_003', CURRENT_DATE(), 300000.0, 100000.0, 400000.0),
('alloc_003', 'org_004', CURRENT_DATE(), 250000.0, 75000.0, 325000.0),
('alloc_004', 'org_005', CURRENT_DATE(), 200000.0, 60000.0, 260000.0);
```


## Step 5: Application Configuration

### Create Configuration Files

**`config/dashboard_config.yaml`:**

```yaml
dashboard:
  name: "Enterprise Cost Dashboard"
  version: "1.0.0"
  refresh_interval: 30
  timezone: "UTC"
  theme: "bootstrap"
  debug: false

databricks:
  connection_pool_size: 10
  query_timeout: 300
  retry_attempts: 3
  retry_delay: 5

organizational_structure:
  default_hierarchy_type: "functional"
  max_depth: 8
  cost_center_format: "CC-{:04d}"
  
cost_allocation:
  default_method: "step_down"
  minimum_allocation: 100.0
  escalation_rates:
    annual: 3.5
    quarterly: 0.875
  
calculation_rules:
  direct_costs:
    - type: "salary"
      allocation_driver: "headcount"
      frequency: "monthly"
    - type: "equipment"
      allocation_driver: "usage_hours"
      frequency: "daily"
  
  indirect_costs:
    - type: "facilities"
      allocation_driver: "square_footage"
      frequency: "monthly"
    - type: "utilities"
      allocation_driver: "headcount"
      frequency: "monthly"

user_interface:
  default_date_range: "90d"
  chart_themes:
    - "plotly_white"
    - "plotly_dark"
    - "seaborn"
  export_formats: ["xlsx", "csv", "pdf"]

logging:
  level: "INFO"
  format: "json"
  handlers:
    - "console"
    - "file"
  file_path: "logs/dashboard.log"
```


### Create Main Application File

**`app.py`:**

```python
import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
import logging
from components.config_manager import ConfigManager
from components.databricks_client import DatabricksClient, RealTimeDataStream
from components.cost_calculator import CostCalculatorEngine, CustomCostCalculator

# Load environment variables
load_dotenv()

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.load_configuration()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize core components
try:
    db_client = DatabricksClient()
    cost_engine = CostCalculatorEngine()
    custom_calculator = CustomCostCalculator(cost_engine)
    data_stream = RealTimeDataStream(db_client)
    logger.info("Core components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize core components: {e}")
    raise

# Initialize Dash app
app = dash.Dash(
    __name__, 
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        dbc.icons.FONT_AWESOME,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ],
    suppress_callback_exceptions=True,
    title="Enterprise Cost Dashboard"
)

# Server reference for deployment
server = app.server

# Global navigation component
def create_navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("ðŸ“Š Dashboard", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("ðŸ“ˆ Analysis", href="/cost-analysis", active="exact")),
            dbc.NavItem(dbc.NavLink("ðŸ§® Calculator", href="/calculator", active="exact")),
            dbc.NavItem(dbc.NavLink("ðŸ¢ Hierarchy", href="/hierarchy", active="exact")),
            dbc.NavItem(dbc.NavLink("âš™ï¸ Settings", href="/settings", active="exact")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("ðŸ“Š Export Data", href="#"),
                    dbc.DropdownMenuItem("ðŸ“ Import Rules", href="#"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("â“ Help", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand="ðŸ¢ Enterprise Cost Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        fluid=True,
    )

# Main app layout
app.layout = dbc.Container([
    create_navbar(),
    html.Br(),
    
    # Connection status indicator
    dbc.Alert(
        [
            html.I(className="fas fa-circle me-2"),
            html.Span("Connected to Databricks", id="connection-status")
        ],
        color="success",
        className="d-flex align-items-center",
        style={"margin-bottom": "20px"}
    ),
    
    # Page content container
    dash.page_container,
    
    # Real-time update components
    dcc.Interval(
        id='realtime-interval',
        interval=config.get('dashboard', {}).get('refresh_interval', 30) * 1000,
        n_intervals=0
    ),
    
    # Store components for shared data
    dcc.Store(id='org-hierarchy-store'),
    dcc.Store(id='cost-rules-store'),
    dcc.Store(id='realtime-data-store'),
    dcc.Store(id='user-preferences-store'),
    
], fluid=True)

# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background-color: #f8f9fa;
            }
            .navbar-brand {
                font-weight: 600;
            }
            .card {
                box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                border: 1px solid rgba(0, 0, 0, 0.125);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    # Start real-time data streaming
    try:
        data_stream.start_streaming()
        logger.info("Real-time data streaming started")
    except Exception as e:
        logger.warning(f"Could not start real-time streaming: {e}")
    
    # Run application
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 8050))
    
    logger.info(f"Starting dashboard on {host}:{port} (debug={debug_mode})")
    app.run_server(
        debug=debug_mode, 
        host=host, 
        port=port,
        dev_tools_hot_reload=debug_mode
    )
```


## Step 6: Create Git Configuration

**`.gitignore`:**

```gitignore
# Environment variables
.env
.env.local
.env.production

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Cache
.cache/
.pytest_cache/
.coverage

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temporary files
temp/
tmp/
```


## Step 7: Testing the Installation

### Run Initial Test

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Test Databricks connection
python test_connection.py

# Run the dashboard
python app.py
```


### Verify Installation

1. **Open your browser** and navigate to `http://localhost:8050`
2. **Check connection status** - Should show "Connected to Databricks"
3. **Test navigation** - Click through different pages
4. **Verify data loading** - Ensure sample data displays correctly

## Step 8: Customization Guide

### Adding Custom Data Sources

**Modify the database client to connect to your specific tables:**

```python
# In components/databricks_client.py
def get_your_custom_data(self, filters=None):
    """Customize this method for your specific data structure"""
    query = """
    SELECT 
        your_org_column,
        your_cost_column,
        your_date_column
    FROM your_schema.your_table
    WHERE your_conditions
    """
    return self.execute_query(query, filters)
```


### Customizing Cost Calculation Rules

**Add your business-specific calculation logic:**

```python
# In components/cost_calculator.py
class YourCustomCalculator(CostCalculatorEngine):
    def your_custom_allocation_method(self, data, rules):
        """Implement your organization's specific allocation logic"""
        # Your custom implementation here
        pass
```


### Customizing Dashboard Layout

**Modify page layouts in the `pages/` directory:**

```python
# In pages/your_custom_page.py
import dash
from dash import html, dcc, callback

dash.register_page(__name__, path='/your-page', name='Your Page')

def layout():
    return html.Div([
        html.H1("Your Custom Page"),
        # Your custom components
    ])
```


## Step 9: Production Deployment

### Docker Deployment

**Create `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8050

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "4", "--timeout", "120", "app:server"]
```

**Create `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8050:8050"
    environment:
      - ENVIRONMENT=production
      - DEBUG=False
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```


### Build and Deploy

```bash
# Build Docker image
docker build -t cost-dashboard .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f dashboard
```


### Cloud Deployment Options

**For AWS/Azure/GCP deployment:**

1. **Containerized Deployment** [^4_11][^4_12]:
    - Use AWS ECS, Azure Container Instances, or Google Cloud Run
    - Configure environment variables in cloud console
    - Set up load balancing and auto-scaling
2. **Platform-as-a-Service**:
    - Deploy to Heroku, PythonAnywhere, or similar platforms
    - Follow platform-specific deployment guides [^4_13][^4_14]
3. **Kubernetes Deployment**:
    - Create Kubernetes manifests
    - Deploy to managed Kubernetes services

## Step 10: Monitoring and Maintenance

### Setup Logging

**Configure structured logging in `components/logger.py`:**

```python
import structlog
import logging.config

def setup_logging():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json"
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "logs/dashboard.log",
                "formatter": "json"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "INFO"
            }
        }
    })
```


### Health Check Endpoint

**Add health monitoring:**

```python
# Add to app.py
from flask import jsonify

@server.route('/health')
def health_check():
    try:
        # Test database connection
        db_client.execute_query("SELECT 1")
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
```


### Backup and Recovery

**Setup automated data backups:**

```bash
# Create backup script
#!/bin/bash
# backup_config.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups"

mkdir -p $BACKUP_DIR

# Backup configuration
cp config/dashboard_config.yaml $BACKUP_DIR/config_$DATE.yaml
cp .env $BACKUP_DIR/env_$DATE.backup

# Export custom rules (implement in your app)
python -c "
from components.config_manager import ConfigManager
import json
config = ConfigManager()
rules = config.get_all_custom_rules()
with open('$BACKUP_DIR/rules_$DATE.json', 'w') as f:
    json.dump(rules, f, indent=2)
"

echo "Backup completed: $DATE"
```


## Troubleshooting Guide

### Common Issues and Solutions

**Connection Issues:**

```bash
# Test Databricks connectivity
curl -H "Authorization: Bearer $DATABRICKS_TOKEN" \
     "https://$DATABRICKS_SERVER_HOSTNAME/api/2.0/clusters/list"
```

**Dependencies Issues:**

```bash
# Clear cache and reinstall
pip cache purge
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

**Performance Issues:**

- Increase connection pool size
- Enable query result caching
- Optimize database queries
- Use Redis for session storage

This comprehensive setup guide provides everything needed to install, configure, and deploy your Enterprise Cost Dashboard with Databricks integration. The modular architecture allows for easy customization while maintaining production-ready standards [^4_15][^4_16][^4_17].
<span style="display:none">[^4_18][^4_19][^4_20][^4_21][^4_22][^4_23][^4_24][^4_25][^4_26][^4_27][^4_28][^4_29][^4_30][^4_31][^4_32][^4_33][^4_34][^4_35][^4_36][^4_37][^4_38][^4_39][^4_40]</span>

<div align="center">â‚</div>

[^4_1]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

[^4_2]: https://frankcorso.dev/setting-up-python-environment-venv-requirements.html

[^4_3]: https://www.geeksforgeeks.org/python/how-to-create-requirements-txt-file-in-python/

[^4_4]: https://www.freecodecamp.org/news/python-requirementstxt-explained/

[^4_5]: https://learn.microsoft.com/en-us/azure/databricks/dev-tools/python-sql-connector

[^4_6]: https://pypi.org/project/databricks-sql-connector/

[^4_7]: https://stackoverflow.com/questions/78728083/how-do-i-configure-python-dash-app-to-read-from-env-development-and-env-produc

[^4_8]: https://docs.catalyst.zoho.com/en/slate/help/environment-variables/

[^4_9]: https://learn.microsoft.com/en-us/azure/databricks/schemas/create-schema

[^4_10]: https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-schema

[^4_11]: https://ekimetrics.github.io/blog/dash-deployment/

[^4_12]: https://community.plotly.com/t/running-dash-app-in-docker-container/16067

[^4_13]: https://towardsdatascience.com/deploying-your-dash-app-to-heroku-the-magical-guide-39bd6a0c586c/

[^4_14]: https://www.youtube.com/watch?v=Gv910_b5ID0

[^4_15]: https://dash.plotly.com/deployment

[^4_16]: https://community.plotly.com/t/can-i-use-dash-plotly-in-production-environment/13541

[^4_17]: https://community.plotly.com/t/production-and-development-enviroments/21348

[^4_18]: https://community.plotly.com/t/how-to-deploy-dash-app-internally-in-local-network/61312

[^4_19]: https://python.plainenglish.io/building-a-dockerised-structured-multipage-plotly-dash-app-3fb53c8703a0

[^4_20]: https://docs.databricks.com/aws/en/dev-tools/python-sql-connector

[^4_21]: https://docs.docker.com/get-started/workshop/02_our_app/

[^4_22]: https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/sql-server-source-setup

[^4_23]: https://github.com/ROpdam/docker-dash-example

[^4_24]: https://learn.microsoft.com/en-us/power-query/connectors/databricks

[^4_25]: https://www.codearmo.com/python-tutorial/making-plotly-dash-app-redis-and-docker-compose

[^4_26]: https://docs.databricks.com/aws/en/dev-tools/sql-drivers-tools

[^4_27]: https://towardsdatascience.com/azure-container-app-a-data-analytics-webapp-with-python-flask-plotly-dash-docker-869686a558cc/

[^4_28]: https://stackoverflow.com/questions/41427500/creating-a-virtualenv-with-preinstalled-packages-as-in-requirements-txt

[^4_29]: https://openclassrooms.com/en/courses/6900846-set-up-a-python-environment/6990546-manage-virtual-environments-using-requirements-files

[^4_30]: https://www.reddit.com/r/PythonLearning/comments/1d57qpu/where_to_make_requirementstxt_for_virtual/

[^4_31]: https://cloud.google.com/run/docs/configuring/services/environment-variables

[^4_32]: https://vercel.com/docs/environment-variables

[^4_33]: https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-schema

[^4_34]: https://www.youtube.com/watch?v=h8bt4RvE7zM

[^4_35]: https://www.reddit.com/r/dotnet/comments/1ideclf/how_are_environment_variables_usually_implemented/

[^4_36]: https://docs.databricks.com/aws/en/schemas/create-schema

[^4_37]: https://docs.python.org/3/library/venv.html

[^4_38]: https://stackoverflow.com/questions/76051945/best-practices-for-accessing-environment-variables-in-production-repo

[^4_39]: https://www.youtube.com/watch?v=x-mt4r0yq9E

[^4_40]: https://learn.microsoft.com/en-us/azure/app-service/reference-app-settings


---

# Parquet to SQLite Development Setup Guide

This guide provides scripts to convert your parquet files to SQLite for development, then seamlessly switch to Databricks when ready.

## Step 1: Parquet to SQLite Conversion Script

### Create Data Migration Utilities

**`scripts/parquet_to_sqlite.py`:**

```python
import os
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParquetToSQLiteConverter:
    def __init__(self, sqlite_db_path: str = "data/development.db"):
        self.sqlite_db_path = sqlite_db_path
        self.connection = None
        self.schema_mapping = self._get_databricks_to_sqlite_mapping()
        
        # Ensure data directory exists
        Path(sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        
    def _get_databricks_to_sqlite_mapping(self) -> Dict[str, str]:
        """Map Databricks/Parquet types to SQLite types"""
        return {
            'string': 'TEXT',
            'int64': 'INTEGER',
            'int32': 'INTEGER',
            'float64': 'REAL',
            'float32': 'REAL',
            'bool': 'INTEGER',  # SQLite uses INTEGER for boolean
            'datetime64[ns]': 'TIMESTAMP',
            'timestamp[ns]': 'TIMESTAMP',
            'date32[day]': 'DATE',
            'decimal': 'DECIMAL',
            'binary': 'BLOB'
        }
    
    def connect(self):
        """Create SQLite connection with optimized settings"""
        self.connection = sqlite3.connect(
            self.sqlite_db_path,
            check_same_thread=False,
            timeout=30.0
        )
        
        # Optimize SQLite settings for development
        self.connection.executescript("""
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;
            PRAGMA mmap_size = 268435456; -- 256MB
            PRAGMA cache_size = 10000;
        """)
        
        logger.info(f"Connected to SQLite database: {self.sqlite_db_path}")
    
    def close(self):
        """Close SQLite connection"""
        if self.connection:
            self.connection.close()
            logger.info("SQLite connection closed")
    
    def analyze_parquet_schema(self, parquet_path: str) -> Dict[str, Any]:
        """Analyze parquet file schema and infer SQLite schema"""
        try:
            # Read parquet file metadata
            parquet_file = pq.ParquetFile(parquet_path)
            schema = parquet_file.schema
            
            # Get sample data for better type inference
            df_sample = pd.read_parquet(parquet_path, nrows=1000)
            
            schema_info = {
                'file_path': parquet_path,
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': len(schema),
                'columns': {},
                'sqlite_ddl': None
            }
            
            # Analyze each column
            for i, field in enumerate(schema):
                column_name = field.name
                parquet_type = str(field.type)
                
                # Infer better type from pandas sample
                if column_name in df_sample.columns:
                    pandas_dtype = str(df_sample[column_name].dtype)
                    is_nullable = df_sample[column_name].isnull().any()
                else:
                    pandas_dtype = parquet_type
                    is_nullable = True
                
                # Map to SQLite type
                sqlite_type = self._map_to_sqlite_type(parquet_type, pandas_dtype)
                
                schema_info['columns'][column_name] = {
                    'position': i,
                    'parquet_type': parquet_type,
                    'pandas_type': pandas_dtype,
                    'sqlite_type': sqlite_type,
                    'nullable': is_nullable
                }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error analyzing parquet schema for {parquet_path}: {e}")
            raise
    
    def _map_to_sqlite_type(self, parquet_type: str, pandas_type: str) -> str:
        """Map parquet/pandas types to SQLite types"""
        # Check pandas type first (more accurate)
        if 'int' in pandas_type:
            return 'INTEGER'
        elif 'float' in pandas_type:
            return 'REAL'
        elif 'bool' in pandas_type:
            return 'INTEGER'
        elif 'datetime' in pandas_type or 'timestamp' in pandas_type:
            return 'TIMESTAMP'
        elif 'object' in pandas_type:
            return 'TEXT'
        
        # Fallback to parquet type mapping
        for key, value in self.schema_mapping.items():
            if key in parquet_type.lower():
                return value
        
        # Default to TEXT
        return 'TEXT'
    
    def create_sqlite_schema(self):
        """Create SQLite schema that mirrors Databricks structure"""
        if not self.connection:
            self.connect()
        
        # Create main cost analysis tables
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS organizations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                level INTEGER NOT NULL,
                cost_center_code TEXT,
                structure_type TEXT DEFAULT 'functional',
                direct_cost_rate REAL DEFAULT 0.0,
                indirect_cost_rate REAL DEFAULT 0.0,
                active INTEGER DEFAULT 1,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES organizations(id)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS cost_allocations (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                transaction_date DATE NOT NULL,
                direct_costs REAL DEFAULT 0.0,
                indirect_costs REAL DEFAULT 0.0,
                allocated_costs REAL DEFAULT 0.0,
                cost_pool_id TEXT,
                allocation_method TEXT,
                transaction_id TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (org_id) REFERENCES organizations(id)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS cost_categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                category_type TEXT NOT NULL,
                allocation_driver TEXT,
                active INTEGER DEFAULT 1,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES cost_categories(id)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS allocation_drivers (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                driver_type TEXT NOT NULL,
                driver_value REAL NOT NULL,
                measurement_date DATE NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (org_id) REFERENCES organizations(id)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS cost_rules (
                id TEXT PRIMARY KEY,
                rule_name TEXT NOT NULL,
                allocation_method TEXT NOT NULL,
                driver_type TEXT NOT NULL,
                source_cost_center TEXT,
                target_cost_centers TEXT,
                allocation_percentage REAL,
                minimum_allocation REAL,
                maximum_allocation REAL,
                escalation_rate REAL DEFAULT 0.0,
                effective_date DATE NOT NULL,
                expiry_date DATE,
                active INTEGER DEFAULT 1,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        # Execute DDL statements
        for ddl in ddl_statements:
            try:
                self.connection.execute(ddl)
                logger.info("Created table from DDL")
            except sqlite3.Error as e:
                logger.error(f"Error creating table: {e}")
                raise
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_organizations_parent ON organizations(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_organizations_level ON organizations(level)",
            "CREATE INDEX IF NOT EXISTS idx_cost_allocations_org ON cost_allocations(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_cost_allocations_date ON cost_allocations(transaction_date)",
            "CREATE INDEX IF NOT EXISTS idx_allocation_drivers_org ON allocation_drivers(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_allocation_drivers_type ON allocation_drivers(driver_type)"
        ]
        
        for index in indexes:
            try:
                self.connection.execute(index)
            except sqlite3.Error as e:
                logger.warning(f"Could not create index: {e}")
        
        self.connection.commit()
        logger.info("SQLite schema created successfully")
    
    def load_parquet_to_table(self, parquet_path: str, table_name: str, 
                             chunk_size: int = 10000, if_exists: str = 'replace'):
        """Load parquet file data into SQLite table"""
        if not self.connection:
            self.connect()
        
        try:
            logger.info(f"Loading {parquet_path} into table {table_name}")
            
            # Read parquet file in chunks for memory efficiency
            parquet_file = pq.ParquetFile(parquet_path)
            total_rows = parquet_file.metadata.num_rows
            
            processed_rows = 0
            
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                # Convert to pandas DataFrame
                df_chunk = batch.to_pandas()
                
                # Handle data type conversions
                df_chunk = self._prepare_dataframe_for_sqlite(df_chunk)
                
                # Load chunk into SQLite
                df_chunk.to_sql(
                    table_name, 
                    self.connection, 
                    if_exists=if_exists if processed_rows == 0 else 'append',
                    index=False,
                    method='multi'
                )
                
                processed_rows += len(df_chunk)
                logger.info(f"Loaded {processed_rows}/{total_rows} rows ({processed_rows/total_rows*100:.1f}%)")
            
            self.connection.commit()
            logger.info(f"Successfully loaded {processed_rows} rows into {table_name}")
            
        except Exception as e:
            logger.error(f"Error loading parquet file {parquet_path}: {e}")
            self.connection.rollback()
            raise
    
    def _prepare_dataframe_for_sqlite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for SQLite insertion"""
        df_prepared = df.copy()
        
        # Convert boolean columns to integers (SQLite compatibility)
        for col in df_prepared.columns:
            if df_prepared[col].dtype == 'bool':
                df_prepared[col] = df_prepared[col].astype(int)
        
        # Handle datetime columns
        for col in df_prepared.columns:
            if pd.api.types.is_datetime64_any_dtype(df_prepared[col]):
                df_prepared[col] = df_prepared[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle None/NaN values appropriately
        df_prepared = df_prepared.where(pd.notnull(df_prepared), None)
        
        return df_prepared
    
    def create_table_from_parquet(self, parquet_path: str, table_name: str):
        """Create SQLite table from parquet file schema and load data"""
        if not self.connection:
            self.connect()
        
        # Analyze parquet schema
        schema_info = self.analyze_parquet_schema(parquet_path)
        
        # Generate CREATE TABLE statement
        columns = []
        for col_name, col_info in schema_info['columns'].items():
            nullable = "" if col_info['nullable'] else " NOT NULL"
            columns.append(f"    {col_name} {col_info['sqlite_type']}{nullable}")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
{',\n'.join(columns)}
        )
        """
        
        logger.info(f"Creating table {table_name} with schema:\n{create_table_sql}")
        
        # Create table
        self.connection.execute(create_table_sql)
        self.connection.commit()
        
        # Load data
        self.load_parquet_to_table(parquet_path, table_name)
    
    def bulk_convert_parquets(self, parquet_directory: str, table_mapping: Optional[Dict[str, str]] = None):
        """Convert multiple parquet files to SQLite tables"""
        parquet_dir = Path(parquet_directory)
        
        if not parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {parquet_directory}")
        
        parquet_files = list(parquet_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {parquet_directory}")
            return
        
        logger.info(f"Found {len(parquet_files)} parquet files to convert")
        
        for parquet_file in parquet_files:
            # Determine table name
            if table_mapping and str(parquet_file) in table_mapping:
                table_name = table_mapping[str(parquet_file)]
            else:
                table_name = parquet_file.stem  # Use filename without extension
            
            try:
                self.create_table_from_parquet(str(parquet_file), table_name)
            except Exception as e:
                logger.error(f"Failed to convert {parquet_file}: {e}")
                continue
    
    def generate_sample_data(self):
        """Generate sample data for testing (if parquet files don't exist)"""
        if not self.connection:
            self.connect()
        
        sample_data = {
            'organizations': [
                ('org_001', 'Enterprise Corp', None, 1, 'CC-0001', 'functional', 0.0, 0.0, 1),
                ('org_002', 'Engineering Division', 'org_001', 2, 'CC-0002', 'functional', 0.0, 0.0, 1),
                ('org_003', 'Sales Division', 'org_001', 2, 'CC-0003', 'functional', 0.0, 0.0, 1),
                ('org_004', 'Software Engineering', 'org_002', 3, 'CC-0004', 'functional', 0.0, 0.0, 1),
                ('org_005', 'Data Engineering', 'org_002', 3, 'CC-0005', 'functional', 0.0, 0.0, 1),
            ],
            'cost_categories': [
                ('cat_001', 'Direct Costs', None, 'direct', 'direct', 1),
                ('cat_002', 'Personnel', 'cat_001', 'direct', 'headcount', 1),
                ('cat_003', 'Equipment', 'cat_001', 'direct', 'usage_hours', 1),
                ('cat_004', 'Indirect Costs', None, 'indirect', 'allocated', 1),
                ('cat_005', 'Facilities', 'cat_004', 'indirect', 'square_footage', 1),
            ],
            'allocation_drivers': [
                ('drv_001', 'org_002', 'headcount', 150.0, '2025-09-29'),
                ('drv_002', 'org_002', 'revenue', 5000000.0, '2025-09-29'),
                ('drv_003', 'org_003', 'headcount', 80.0, '2025-09-29'),
                ('drv_004', 'org_003', 'revenue', 12000000.0, '2025-09-29'),
            ],
            'cost_allocations': [
                ('alloc_001', 'org_002', '2025-09-29', 500000.0, 150000.0, 650000.0, 'pool_001', 'direct', 'txn_001'),
                ('alloc_002', 'org_003', '2025-09-29', 300000.0, 100000.0, 400000.0, 'pool_002', 'direct', 'txn_002'),
                ('alloc_003', 'org_004', '2025-09-29', 250000.0, 75000.0, 325000.0, 'pool_003', 'step_down', 'txn_003'),
            ]
        }
        
        # Insert sample data
        for table_name, rows in sample_data.items():
            # Get column count for placeholders
            if rows:
                placeholders = ','.join(['?' for _ in range(len(rows[^5_0]))])
                insert_sql = f"INSERT OR REPLACE INTO {table_name} VALUES ({placeholders})"
                
                try:
                    self.connection.executemany(insert_sql, rows)
                    logger.info(f"Inserted {len(rows)} sample rows into {table_name}")
                except sqlite3.Error as e:
                    logger.error(f"Error inserting sample data into {table_name}: {e}")
        
        self.connection.commit()
        logger.info("Sample data generation completed")

def main():
    parser = argparse.ArgumentParser(description='Convert Parquet files to SQLite database')
    parser.add_argument('--parquet-dir', type=str, help='Directory containing parquet files')
    parser.add_argument('--sqlite-db', type=str, default='data/development.db', 
                       help='SQLite database path')
    parser.add_argument('--create-schema', action='store_true', 
                       help='Create standard cost analysis schema')
    parser.add_argument('--generate-sample', action='store_true',
                       help='Generate sample data for testing')
    parser.add_argument('--table-mapping', type=str, 
                       help='JSON file mapping parquet files to table names')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = ParquetToSQLiteConverter(args.sqlite_db)
    
    try:
        converter.connect()
        
        # Create schema if requested
        if args.create_schema:
            converter.create_sqlite_schema()
        
        # Convert parquet files if directory provided
        if args.parquet_dir:
            table_mapping = None
            if args.table_mapping:
                with open(args.table_mapping, 'r') as f:
                    table_mapping = json.load(f)
            
            converter.bulk_convert_parquets(args.parquet_dir, table_mapping)
        
        # Generate sample data if requested
        if args.generate_sample:
            converter.generate_sample_data()
        
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise
    finally:
        converter.close()

if __name__ == "__main__":
    main()
```


## Step 2: Database Abstraction Layer

Create a database abstraction layer that works with both SQLite and Databricks [^5_1][^5_2]:

**`components/database_manager.py`:**

```python
import os
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine
from databricks import sql as databricks_sql
from dotenv import load_dotenv

load_dotenv()

class DatabaseType(Enum):
    SQLITE = "sqlite"
    DATABRICKS = "databricks"

class DatabaseManager:
    """Unified database interface supporting SQLite and Databricks"""
    
    def __init__(self, db_type: DatabaseType = None):
        self.db_type = db_type or self._determine_database_type()
        self.engine: Optional[Engine] = None
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
        # Connection parameters
        self.connection_params = self._get_connection_params()
        
        # Initialize connection
        self._initialize_connection()
    
    def _determine_database_type(self) -> DatabaseType:
        """Automatically determine database type from environment"""
        environment = os.getenv('ENVIRONMENT', 'development').lower()
        
        # Use SQLite for development, Databricks for production
        if environment in ['development', 'dev', 'local']:
            return DatabaseType.SQLITE
        elif os.getenv('DATABRICKS_TOKEN'):
            return DatabaseType.DATABRICKS
        else:
            # Default to SQLite if no Databricks token
            return DatabaseType.SQLITE
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters based on database type"""
        if self.db_type == DatabaseType.SQLITE:
            return {
                'database_path': os.getenv('SQLITE_DATABASE_PATH', 'data/development.db'),
                'timeout': 30.0,
                'check_same_thread': False
            }
        elif self.db_type == DatabaseType.DATABRICKS:
            return {
                'server_hostname': os.getenv('DATABRICKS_SERVER_HOSTNAME'),
                'http_path': os.getenv('DATABRICKS_HTTP_PATH'),
                'access_token': os.getenv('DATABRICKS_TOKEN'),
                'catalog': os.getenv('DEFAULT_CATALOG', 'main'),
                'schema': os.getenv('DEFAULT_SCHEMA', 'cost_analysis')
            }
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                self._connect_sqlite()
            elif self.db_type == DatabaseType.DATABRICKS:
                self._connect_databricks()
            
            self.logger.info(f"Connected to {self.db_type.value} database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.db_type.value}: {e}")
            raise
    
    def _connect_sqlite(self):
        """Connect to SQLite database"""
        db_path = self.connection_params['database_path']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create SQLAlchemy engine
        connection_string = f"sqlite:///{db_path}"
        self.engine = create_engine(
            connection_string,
            connect_args={
                'timeout': self.connection_params['timeout'],
                'check_same_thread': self.connection_params['check_same_thread']
            },
            pool_pre_ping=True,
            echo=False
        )
        
        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    
    def _connect_databricks(self):
        """Connect to Databricks"""
        # Validate required parameters
        required_params = ['server_hostname', 'http_path', 'access_token']
        for param in required_params:
            if not self.connection_params.get(param):
                raise ValueError(f"Missing required parameter: {param}")
        
        # Create SQLAlchemy engine for Databricks
        connection_string = (
            f"databricks://token:{self.connection_params['access_token']}@"
            f"{self.connection_params['server_hostname']}:443"
            f"{self.connection_params['http_path']}"
        )
        
        self.engine = create_engine(
            connection_string,
            connect_args={
                'http_path': self.connection_params['http_path'],
                'server_hostname': self.connection_params['server_hostname']
            }
        )
        
        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text("SELECT current_timestamp()"))
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return df
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self.logger.error(f"Query: {query}")
            raise
    
    def execute_non_query(self, query: str, params: Optional[Dict] = None) -> int:
        """Execute non-query statement (INSERT, UPDATE, DELETE)"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result.rowcount
                
        except Exception as e:
            self.logger.error(f"Non-query execution failed: {e}")
            self.logger.error(f"Query: {query}")
            raise
    
    def bulk_insert(self, table_name: str, data: Union[pd.DataFrame, List[Dict]], 
                   if_exists: str = 'append') -> int:
        """Bulk insert data into table"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Adjust table name for Databricks (include schema)
            full_table_name = self._get_full_table_name(table_name)
            
            df.to_sql(
                full_table_name.split('.')[-1],  # Just table name for to_sql
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Bulk insert failed for table {table_name}: {e}")
            raise
    
    def _get_full_table_name(self, table_name: str) -> str:
        """Get fully qualified table name"""
        if self.db_type == DatabaseType.DATABRICKS:
            catalog = self.connection_params.get('catalog', 'main')
            schema = self.connection_params.get('schema', 'cost_analysis')
            return f"{catalog}.{schema}.{table_name}"
        else:
            return table_name
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema information"""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            schema = {}
            for column in columns:
                schema[column['name']] = str(column['type'])
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Failed to get schema for table {table_name}: {e}")
            return {}
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            inspector = inspect(self.engine)
            return table_name in inspector.get_table_names()
        except Exception:
            return False
    
    def get_cost_hierarchy_data(self, org_id: Optional[str] = None, 
                               date_range: Optional[tuple] = None) -> pd.DataFrame:
        """Get hierarchical cost data (compatible with both databases)"""
        
        # Base query that works with both SQLite and Databricks
        if self.db_type == DatabaseType.SQLITE:
            # SQLite recursive CTE
            query = """
            WITH RECURSIVE org_hierarchy AS (
                SELECT 
                    id, name, parent_id, level, cost_center_code,
                    direct_cost_rate, indirect_cost_rate
                FROM organizations
                WHERE (:org_id IS NULL OR id = :org_id)
                UNION ALL
                SELECT 
                    o.id, o.name, o.parent_id, oh.level + 1, o.cost_center_code,
                    o.direct_cost_rate, o.indirect_cost_rate
                FROM organizations o
                INNER JOIN org_hierarchy oh ON o.parent_id = oh.id
            ),
            cost_aggregation AS (
                SELECT 
                    oh.id, oh.name, oh.level, oh.cost_center_code,
                    oh.direct_cost_rate, oh.indirect_cost_rate,
                    COALESCE(SUM(ca.direct_costs), 0) as total_direct_costs,
                    COALESCE(SUM(ca.indirect_costs), 0) as total_indirect_costs,
                    COALESCE(SUM(ca.allocated_costs), 0) as total_allocated_costs,
                    COUNT(ca.id) as transaction_count
                FROM org_hierarchy oh
                LEFT JOIN cost_allocations ca ON oh.id = ca.org_id
                WHERE (:start_date IS NULL OR ca.transaction_date >= :start_date)
                  AND (:end_date IS NULL OR ca.transaction_date <= :end_date)
                GROUP BY oh.id, oh.name, oh.level, oh.cost_center_code,
                         oh.direct_cost_rate, oh.indirect_cost_rate
            )
            SELECT * FROM cost_aggregation
            ORDER BY level, name
            """
        else:
            # Databricks recursive CTE (similar syntax)
            full_org_table = self._get_full_table_name('organizations')
            full_alloc_table = self._get_full_table_name('cost_allocations')
            
            query = f"""
            WITH RECURSIVE org_hierarchy AS (
                SELECT 
                    id, name, parent_id, level, cost_center_code,
                    direct_cost_rate, indirect_cost_rate
                FROM {full_org_table}
                WHERE (:org_id IS NULL OR id = :org_id)
                UNION ALL
                SELECT 
                    o.id, o.name, o.parent_id, oh.level + 1, o.cost_center_code,
                    o.direct_cost_rate, o.indirect_cost_rate
                FROM {full_org_table} o
                INNER JOIN org_hierarchy oh ON o.parent_id = oh.id
            ),
            cost_aggregation AS (
                SELECT 
                    oh.id, oh.name, oh.level, oh.cost_center_code,
                    oh.direct_cost_rate, oh.indirect_cost_rate,
                    COALESCE(SUM(ca.direct_costs), 0) as total_direct_costs,
                    COALESCE(SUM(ca.indirect_costs), 0) as total_indirect_costs,
                    COALESCE(SUM(ca.allocated_costs), 0) as total_allocated_costs,
                    COUNT(ca.id) as transaction_count
                FROM org_hierarchy oh
                LEFT JOIN {full_alloc_table} ca ON oh.id = ca.org_id
                WHERE (:start_date IS NULL OR ca.transaction_date >= :start_date)
                  AND (:end_date IS NULL OR ca.transaction_date <= :end_date)
                GROUP BY oh.id, oh.name, oh.level, oh.cost_center_code,
                         oh.direct_cost_rate, oh.indirect_cost_rate
            )
            SELECT * FROM cost_aggregation
            ORDER BY level, name
            """
        
        # Prepare parameters
        params = {
            'org_id': org_id,
            'start_date': date_range[^5_0] if date_range else None,
            'end_date': date_range[^5_1] if date_range else None
        }
        
        return self.execute_query(query, params)
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info(f"Closed {self.db_type.value} connection")

# Singleton instance for global use
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def switch_to_databricks():
    """Switch to Databricks database"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = DatabaseManager(DatabaseType.DATABRICKS)
    return _db_manager

def switch_to_sqlite():
    """Switch to SQLite database"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = DatabaseManager(DatabaseType.SQLITE)
    return _db_manager
```


## Step 3: Configuration Management

Update your configuration to support both databases:

**`.env` additions:**

```bash
# Database Configuration
ENVIRONMENT=development  # development, production
DATABASE_TYPE=sqlite     # sqlite, databricks

# SQLite Configuration (Development)
SQLITE_DATABASE_PATH=data/development.db

# Databricks Configuration (Production)
DATABRICKS_SERVER_HOSTNAME=your-workspace.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/endpoints/your-endpoint-id  
DATABRICKS_TOKEN=your-personal-access-token
DEFAULT_CATALOG=main
DEFAULT_SCHEMA=cost_analysis
```


## Step 4: Updated Application Code

Modify your main application to use the database abstraction layer:

**`components/databricks_client.py` (Updated):**

```python
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from components.database_manager import get_database_manager, DatabaseType

class UnifiedDatabaseClient:
    """Unified client that works with both SQLite and Databricks"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.logger = logging.getLogger(__name__)
    
    @property
    def database_type(self) -> DatabaseType:
        """Get current database type"""
        return self.db_manager.db_type
    
    def execute_query(self, query: str, params: Optional[List] = None) -> List[tuple]:
        """Execute query with parameter substitution"""
        # Convert positional parameters to named parameters for SQLAlchemy
        if params:
            param_dict = {f'param_{i}': param for i, param in enumerate(params)}
            # Replace ? with :param_0, :param_1, etc.
            for i in range(len(params)):
                query = query.replace('?', f':param_{i}', 1)
        else:
            param_dict = None
        
        df = self.db_manager.execute_query(query, param_dict)
        return [tuple(row) for row in df.itertuples(index=False, name=None)]
    
    def get_cost_hierarchy_data(self, org_id: Optional[str] = None, 
                               date_range: Optional[tuple] = None) -> pd.DataFrame:
        """Get hierarchical cost data"""
        return self.db_manager.get_cost_hierarchy_data(org_id, date_range)
    
    def get_latest_cost_updates(self) -> pd.DataFrame:
        """Get latest cost updates for real-time dashboard"""
        query = """
        SELECT 
            org_id, 
            transaction_date,
            direct_costs,
            indirect_costs,
            allocated_costs
        FROM cost_allocations
        WHERE transaction_date >= date('now', '-7 days')
        ORDER BY transaction_date DESC
        LIMIT 100
        """
        
        # Adjust query for Databricks
        if self.database_type == DatabaseType.DATABRICKS:
            query = query.replace("date('now', '-7 days')", "current_date() - INTERVAL 7 DAYS")
        
        return self.db_manager.execute_query(query)
    
    def get_organization_options(self) -> List[Dict[str, Any]]:
        """Get organization options for dropdowns"""
        query = """
        SELECT DISTINCT 
            id as value, 
            name as label,
            level,
            parent_id
        FROM organizations 
        WHERE active = 1
        ORDER BY level, name
        """
        
        # Adjust for boolean handling
        if self.database_type == DatabaseType.DATABRICKS:
            query = query.replace("active = 1", "active = true")
        
        df = self.db_manager.execute_query(query)
        return df.to_dict('records')

# For backward compatibility, create an alias
DatabricksClient = UnifiedDatabaseClient
```


## Step 5: Migration Scripts

### SQLite to Databricks Migration

**`scripts/migrate_to_databricks.py`:**

```python
import os
import logging
from pathlib import Path
import pandas as pd
from components.database_manager import DatabaseManager, DatabaseType
from databricks import sql as databricks_sql
from dotenv import load_dotenv

load_dotenv()

class SQLiteToDatabricksMigrator:
    def __init__(self):
        self.sqlite_manager = DatabaseManager(DatabaseType.SQLITE)
        self.databricks_manager = DatabaseManager(DatabaseType.DATABRICKS)
        self.logger = logging.getLogger(__name__)
    
    def create_databricks_schema(self):
        """Create Databricks schema from SQLite schema"""
        
        # Schema mapping from SQLite to Databricks types
        type_mapping = {
            'INTEGER': 'BIGINT',
            'REAL': 'DOUBLE',
            'TEXT': 'STRING',
            'BLOB': 'BINARY',
            'TIMESTAMP': 'TIMESTAMP',
            'DATE': 'DATE'
        }
        
        # Get all tables from SQLite
        sqlite_tables = [
            'organizations', 'cost_allocations', 'cost_categories', 
            'allocation_drivers', 'cost_rules'
        ]
        
        for table_name in sqlite_tables:
            if self.sqlite_manager.table_exists(table_name):
                # Get SQLite schema
                schema = self.sqlite_manager.get_table_schema(table_name)
                
                # Convert to Databricks DDL
                columns = []
                for col_name, col_type in schema.items():
                    databricks_type = type_mapping.get(col_type.upper(), 'STRING')
                    columns.append(f"    {col_name} {databricks_type}")
                
                # Create table in Databricks
                full_table_name = self.databricks_manager._get_full_table_name(table_name)
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS {full_table_name} (
{',\n'.join(columns)}
                ) USING DELTA
                """
                
                self.logger.info(f"Creating Databricks table: {full_table_name}")
                self.databricks_manager.execute_non_query(create_sql)
    
    def migrate_table_data(self, table_name: str, batch_size: int = 1000):
        """Migrate data from SQLite table to Databricks"""
        self.logger.info(f"Migrating table: {table_name}")
        
        # Read all data from SQLite
        query = f"SELECT * FROM {table_name}"
        df = self.sqlite_manager.execute_query(query)
        
        if df.empty:
            self.logger.warning(f"No data found in table {table_name}")
            return
        
        # Migrate in batches
        total_rows = len(df)
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Insert batch into Databricks
            self.databricks_manager.bulk_insert(table_name, batch_df, if_exists='append')
            
            self.logger.info(f"Migrated {end_idx}/{total_rows} rows from {table_name}")
        
        self.logger.info(f"Completed migration of {table_name}: {total_rows} rows")
    
    def migrate_all_data(self):
        """Migrate all data from SQLite to Databricks"""
        tables_to_migrate = [
            'organizations',
            'cost_categories', 
            'cost_allocations',
            'allocation_drivers',
            'cost_rules'
        ]
        
        # Create schema first
        self.create_databricks_schema()
        
        # Migrate data table by table
        for table_name in tables_to_migrate:
            if self.sqlite_manager.table_exists(table_name):
                try:
                    self.migrate_table_data(table_name)
                except Exception as e:
                    self.logger.error(f"Failed to migrate {table_name}: {e}")
                    continue
        
        self.logger.info("Migration completed!")
    
    def verify_migration(self):
        """Verify data migration by comparing row counts"""
        tables = ['organizations', 'cost_allocations', 'cost_categories', 
                 'allocation_drivers', 'cost_rules']
        
        verification_results = {}
        
        for table_name in tables:
            try:
                # Count rows in SQLite
                sqlite_count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                sqlite_result = self.sqlite_manager.execute_query(sqlite_count_query)
                sqlite_count = sqlite_result.iloc[^5_0]['count'] if not sqlite_result.empty else 0
                
                # Count rows in Databricks
                full_table_name = self.databricks_manager._get_full_table_name(table_name)
                databricks_count_query = f"SELECT COUNT(*) as count FROM {full_table_name}"
                databricks_result = self.databricks_manager.execute_query(databricks_count_query)
                databricks_count = databricks_result.iloc[^5_0]['count'] if not databricks_result.empty else 0
                
                verification_results[table_name] = {
                    'sqlite_count': sqlite_count,
                    'databricks_count': databricks_count,
                    'match': sqlite_count == databricks_count
                }
                
                self.logger.info(f"{table_name}: SQLite={sqlite_count}, Databricks={databricks_count}, Match={sqlite_count == databricks_count}")
                
            except Exception as e:
                self.logger.error(f"Verification failed for {table_name}: {e}")
                verification_results[table_name] = {'error': str(e)}
        
        return verification_results

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    migrator = SQLiteToDatabricksMigrator()
    
    try:
        # Perform migration
        migrator.migrate_all_data()
        
        # Verify migration
        print("\n=== Migration Verification ===")
        verification_results = migrator.verify_migration()
        
        # Print summary
        all_match = all(result.get('match', False) for result in verification_results.values())
        print(f"\nMigration {'SUCCESS' if all_match else 'FAILED'}")
        
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()
```


## Step 6: Usage Instructions

### For Development with SQLite

```bash
# 1. Convert your parquet files to SQLite
python scripts/parquet_to_sqlite.py \
    --parquet-dir /path/to/your/parquet/files \
    --sqlite-db data/development.db \
    --create-schema

# 2. Set environment for development
export ENVIRONMENT=development
export DATABASE_TYPE=sqlite

# 3. Run your dashboard
python app.py
```


### For Production with Databricks

```bash
# 1. Set environment for production
export ENVIRONMENT=production
export DATABASE_TYPE=databricks

# 2. Migrate data from SQLite to Databricks (one-time)
python scripts/migrate_to_databricks.py

# 3. Run your dashboard
python app.py
```


### Seamless Switching

Your application will automatically switch databases based on the `ENVIRONMENT` variable:

- **Development** (`ENVIRONMENT=development`): Uses SQLite
- **Production** (`ENVIRONMENT=production`): Uses Databricks

You can also force a specific database type using the `DATABASE_TYPE` environment variable.

## Key Benefits of This Approach

1. **No Code Changes Required**: Your dashboard code remains the same [^5_1][^5_3]
2. **Fast Development**: Work with local SQLite for quick iterations [^5_4][^5_5]
3. **Production Ready**: Seamlessly switch to Databricks for production [^5_6][^5_7]
4. **Data Integrity**: Migration scripts ensure data consistency [^5_8][^5_9]
5. **Type Safety**: Schema mapping handles type differences between databases [^5_10][^5_11]

This setup allows you to develop and test your dashboard quickly with local data while maintaining the ability to deploy to production Databricks without any code changes [^5_12][^5_13].
<span style="display:none">[^5_14][^5_15][^5_16][^5_17][^5_18][^5_19][^5_20]</span>

<div align="center">â‚</div>

[^5_1]: https://flask.palletsprojects.com/en/stable/patterns/sqlalchemy/

[^5_2]: https://www.sqlalchemy.org

[^5_3]: https://www.reddit.com/r/Python/comments/6mnp9h/which_python_3_database_abstraction_layer_do_you/

[^5_4]: https://mungingdata.com/sqlite/export-tables-queries-to-csv-parquet/

[^5_5]: https://slingdata.io/articles/export-load-local-parquet-sqlite-sling/

[^5_6]: https://bryteflow.com/databricks-connect-to-sql-server-easy-migration-method/

[^5_7]: https://www.matillion.com/connectors/sqlite/databricks

[^5_8]: https://www.reddit.com/r/dataengineering/comments/153xtyq/schema_migration_for_delta_lake_on_databricks/

[^5_9]: https://community.databricks.com/t5/technical-blog/databricks-schema-versioning-with-flyway-and-liquibase-a-step-by/ba-p/90504

[^5_10]: https://stackoverflow.com/questions/61386337/python-database-abstraction-layers-comparison-examples

[^5_11]: https://stackoverflow.com/questions/49375744/abstraction-layer-on-top-of-sqlalchemy

[^5_12]: https://til.simonwillison.net/duckdb/parquet

[^5_13]: https://www.blog.pythonlibrary.org/2024/05/06/how-to-read-and-write-parquet-files-with-python/

[^5_14]: https://stackoverflow.com/questions/75389288/export-a-sqlite-table-to-apache-parquet-without-creating-a-dataframe

[^5_15]: https://cldellow.com/2018/06/22/sqlite-parquet-vtable.html

[^5_16]: https://www.kaggle.com/code/hugoboia/converting-sqlite-to-parquet-with-duckdb

[^5_17]: https://arxiv.org/html/2502.05311v1

[^5_18]: https://mungingdata.com/sqlite/create-database-load-csv-python/

[^5_19]: https://www.databricks.com/blog/navigating-sql-server-databricks-migration-tips-seamless-transition

[^5_20]: https://stackoverflow.com/questions/78910566/make-a-table-sqlite-in-db-schema-table
