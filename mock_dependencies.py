"""
Mock Dependencies - Standalone replacements for missing packages

This module provides mock implementations for external dependencies
to allow ShadowForge OS to run in environments without full package installs.
"""

import sys
from typing import Any, List, Dict
import random
import math


class MockNumPy:
    """Mock numpy module for quantum computing simulations."""
    
    def __init__(self):
        pass
    
    def array(self, data):
        """Mock numpy array."""
        return data
    
    def zeros(self, shape):
        """Mock zeros array."""
        if isinstance(shape, int):
            return [0.0] * shape
        return [[0.0] * shape[1] for _ in range(shape[0])]
    
    def ones(self, shape):
        """Mock ones array."""
        if isinstance(shape, int):
            return [1.0] * shape
        return [[1.0] * shape[1] for _ in range(shape[0])]
    
    def random(self):
        """Mock random submodule."""
        return MockNumPyRandom()
    
    def linalg(self):
        """Mock linear algebra submodule."""
        return MockNumPyLinalg()
    
    def pi(self):
        """Mock pi constant."""
        return math.pi
    
    def exp(self, x):
        """Mock exponential function."""
        return math.exp(x)
    
    def sin(self, x):
        """Mock sin function."""
        return math.sin(x)
    
    def cos(self, x):
        """Mock cos function."""
        return math.cos(x)
    
    def sqrt(self, x):
        """Mock sqrt function."""
        return math.sqrt(x)


class MockNumPyRandom:
    """Mock numpy.random submodule."""
    
    def beta(self, a, b):
        """Mock beta distribution."""
        return random.betavariate(a, b)
    
    def normal(self, mean=0, std=1):
        """Mock normal distribution."""
        return random.gauss(mean, std)
    
    def uniform(self, low=0, high=1):
        """Mock uniform distribution."""
        return random.uniform(low, high)


class MockNumPyLinalg:
    """Mock numpy.linalg submodule."""
    
    def norm(self, vector):
        """Mock vector norm."""
        return math.sqrt(sum(x*x for x in vector))


class MockCrewAI:
    """Mock CrewAI module."""
    
    class Agent:
        def __init__(self, **kwargs):
            self.role = kwargs.get('role', 'agent')
            self.goal = kwargs.get('goal', 'execute tasks')
            self.backstory = kwargs.get('backstory', 'AI agent')
            self.tools = kwargs.get('tools', [])
    
    class Task:
        def __init__(self, **kwargs):
            self.description = kwargs.get('description', 'task')
            self.agent = kwargs.get('agent')
    
    class Crew:
        def __init__(self, **kwargs):
            self.agents = kwargs.get('agents', [])
            self.tasks = kwargs.get('tasks', [])
        
        def kickoff(self):
            return {"status": "completed", "output": "Mock crew execution"}
    
    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"
    
    class BaseTool:
        def __init__(self, **kwargs):
            self.name = kwargs.get('name', 'tool')
            self.description = kwargs.get('description', 'mock tool')
        
        def _run(self, *args, **kwargs):
            return "Mock tool execution"
        
        def _arun(self, *args, **kwargs):
            return "Mock async tool execution"


# Add needed exports
Agent = MockCrewAI.Agent
Task = MockCrewAI.Task  
Crew = MockCrewAI.Crew
Process = MockCrewAI.Process


class MockPandas:
    """Mock pandas module."""
    
    class DataFrame:
        def __init__(self, data=None, **kwargs):
            self.data = data or {}
        
        def head(self, n=5):
            return self
        
        def describe(self):
            return self
        
        def to_dict(self, orient='records'):
            return [self.data] if isinstance(self.data, dict) else self.data


class MockSklearn:
    """Mock scikit-learn module."""
    
    class ensemble:
        class RandomForestRegressor:
            def __init__(self, **kwargs):
                self.n_estimators = kwargs.get('n_estimators', 100)
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0.5] * len(X) if hasattr(X, '__len__') else [0.5]
    
    class linear_model:
        class LinearRegression:
            def __init__(self, **kwargs):
                pass
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0.5] * len(X) if hasattr(X, '__len__') else [0.5]
    
    class metrics:
        @staticmethod
        def mean_squared_error(y_true, y_pred):
            return 0.1
    
    class preprocessing:
        class StandardScaler:
            def __init__(self, **kwargs):
                pass
            
            def fit(self, X):
                return self
            
            def transform(self, X):
                return X if hasattr(X, '__len__') else [X]
            
            def fit_transform(self, X):
                return self.transform(X)


class MockRequests:
    """Mock requests module."""
    
    class Response:
        def __init__(self, data):
            self.status_code = 200
            self._data = data
        
        def json(self):
            return self._data
        
        def text(self):
            return str(self._data)
    
    @staticmethod
    def get(url, **kwargs):
        return MockRequests.Response({"mock": "data", "url": url})
    
    @staticmethod
    def post(url, **kwargs):
        return MockRequests.Response({"mock": "response", "url": url})


class MockNetworkX:
    """Mock NetworkX module."""
    
    class Graph:
        def __init__(self, **kwargs):
            self.nodes_data = {}
            self.edges_data = []
        
        def add_node(self, node, **attr):
            self.nodes_data[node] = attr
        
        def add_edge(self, u, v, **attr):
            self.edges_data.append((u, v, attr))
        
        def nodes(self):
            return list(self.nodes_data.keys())
        
        def edges(self):
            return [(u, v) for u, v, _ in self.edges_data]
    
    class DiGraph(Graph):
        pass
    
    @staticmethod
    def shortest_path(G, source, target):
        return [source, target]
    
    @staticmethod
    def connected_components(G):
        return [set(G.nodes())]


class MockAiosqlite:
    """Mock aiosqlite module."""
    
    class Connection:
        def __init__(self, database):
            self.database = database
            self._closed = False
        
        async def execute(self, sql, parameters=None):
            if self._closed:
                raise RuntimeError("Connection is closed")
            return MockAiosqlite.Cursor()
        
        async def executemany(self, sql, parameters):
            if self._closed:
                raise RuntimeError("Connection is closed")
            return MockAiosqlite.Cursor()
        
        async def commit(self):
            if self._closed:
                raise RuntimeError("Connection is closed")
            pass
        
        async def close(self):
            self._closed = True
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()
    
    class Cursor:
        def __init__(self):
            self.data = []
        
        async def fetchone(self):
            return None
        
        async def fetchall(self):
            return []
        
        async def fetchmany(self, size=1):
            return []
    
    @staticmethod
    async def connect(database):
        # Always return a new connection to avoid thread issues
        return MockAiosqlite.Connection(database)


class MockFastAPI:
    """Mock FastAPI module."""
    
    class FastAPI:
        def __init__(self, **kwargs):
            self.title = kwargs.get("title", "FastAPI")
            self.description = kwargs.get("description", "")
            self.version = kwargs.get("version", "0.1.0")
            self.routes = []
            self.middleware = []
        
        def get(self, path, **kwargs):
            def decorator(func):
                self.routes.append(("GET", path, func))
                return func
            return decorator
        
        def post(self, path, **kwargs):
            def decorator(func):
                self.routes.append(("POST", path, func))
                return func
            return decorator
        
        def put(self, path, **kwargs):
            def decorator(func):
                self.routes.append(("PUT", path, func))
                return func
            return decorator
        
        def delete(self, path, **kwargs):
            def decorator(func):
                self.routes.append(("DELETE", path, func))
                return func
            return decorator
        
        def add_middleware(self, middleware_class, **kwargs):
            self.middleware.append((middleware_class, kwargs))
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            self.status_code = status_code
            self.detail = detail
    
    class Request:
        def __init__(self):
            self.headers = {}
            self.query_params = {}
            self.path_params = {}
    
    @staticmethod
    def Depends(dependency):
        return dependency
    
    class middleware:
        class cors:
            class CORSMiddleware:
                def __init__(self, app, **kwargs):
                    self.app = app
                    self.allow_origins = kwargs.get("allow_origins", [])
                    self.allow_credentials = kwargs.get("allow_credentials", False)
                    self.allow_methods = kwargs.get("allow_methods", ["GET"])
                    self.allow_headers = kwargs.get("allow_headers", [])
    
    class security:
        class HTTPBearer:
            def __init__(self, **kwargs):
                self.auto_error = kwargs.get("auto_error", True)
        
        class HTTPAuthorizationCredentials:
            def __init__(self, scheme="Bearer", credentials="mock_token"):
                self.scheme = scheme
                self.credentials = credentials


class MockUvicorn:
    """Mock Uvicorn module."""
    
    @staticmethod
    def run(app, host="127.0.0.1", port=8000, **kwargs):
        print(f"Mock Uvicorn server running on {host}:{port}")
        return True


class MockLangChain:
    """Mock LangChain module."""
    
    class BaseTool:
        def __init__(self, **kwargs):
            self.name = kwargs.get('name', 'tool')
            self.description = kwargs.get('description', 'mock tool')
        
        def _run(self, *args, **kwargs):
            return "Mock tool execution"
        
        def _arun(self, *args, **kwargs):
            return "Mock async tool execution"
    
    class Ollama:
        def __init__(self, **kwargs):
            self.model = kwargs.get('model', 'llama2')
            self.temperature = kwargs.get('temperature', 0.7)
        
        def generate(self, prompt):
            return f"Mock LLM response to: {prompt[:50]}..."
    
    class ChatOpenAI:
        def __init__(self, **kwargs):
            self.model = kwargs.get('model', 'gpt-3.5-turbo')
            self.temperature = kwargs.get('temperature', 0.7)
        
        def generate(self, prompt):
            return f"Mock OpenAI response to: {prompt[:50]}..."
    
    class Tool:
        def __init__(self, **kwargs):
            self.name = kwargs.get('name', 'tool')
            self.description = kwargs.get('description', 'mock tool')
            self.func = kwargs.get('func', lambda x: f"Mock tool result: {x}")
        
        def run(self, input_data):
            return self.func(input_data)


# Install mock modules
def install_mock_dependencies():
    """Install mock dependencies into sys.modules."""
    
    # Mock numpy
    np = MockNumPy()
    np.random = MockNumPyRandom()
    np.linalg = MockNumPyLinalg()
    np.pi = math.pi
    sys.modules['numpy'] = np
    sys.modules['np'] = np
    
    # Mock CrewAI
    crewai = MockCrewAI()
    crewai.Agent = MockCrewAI.Agent
    crewai.Task = MockCrewAI.Task
    crewai.Crew = MockCrewAI.Crew
    crewai.Process = MockCrewAI.Process
    
    # Create tools submodule
    tools = type('tools', (), {})()
    tools.BaseTool = MockCrewAI.BaseTool
    crewai.tools = tools
    
    sys.modules['crewai'] = crewai
    sys.modules['crewai.tools'] = tools
    
    # Mock LangChain
    langchain = MockLangChain()
    
    # Create submodules
    llms = type('llms', (), {})()
    llms.Ollama = MockLangChain.Ollama
    langchain.llms = llms
    
    tools = type('tools', (), {})()
    tools.BaseTool = MockLangChain.BaseTool
    tools.Tool = MockLangChain.Tool
    langchain.tools = tools
    
    sys.modules['langchain'] = langchain
    sys.modules['langchain.llms'] = llms
    sys.modules['langchain.tools'] = tools
    sys.modules['langchain_core'] = langchain
    sys.modules['langchain_core.tools'] = tools
    
    # Mock langchain_openai
    langchain_openai = type('langchain_openai', (), {})()
    langchain_openai.ChatOpenAI = MockLangChain.ChatOpenAI
    sys.modules['langchain_openai'] = langchain_openai
    
    # Mock pandas
    pd = MockPandas()
    pd.DataFrame = MockPandas.DataFrame
    sys.modules['pandas'] = pd
    
    # Mock sklearn
    sklearn = MockSklearn()
    sklearn.ensemble = MockSklearn.ensemble
    sklearn.linear_model = MockSklearn.linear_model
    sklearn.metrics = MockSklearn.metrics
    sklearn.preprocessing = MockSklearn.preprocessing
    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.ensemble'] = MockSklearn.ensemble
    sys.modules['sklearn.linear_model'] = MockSklearn.linear_model
    sys.modules['sklearn.metrics'] = MockSklearn.metrics
    sys.modules['sklearn.preprocessing'] = MockSklearn.preprocessing
    
    # Mock requests
    requests = MockRequests()
    requests.get = MockRequests.get
    requests.post = MockRequests.post
    requests.Response = MockRequests.Response
    sys.modules['requests'] = requests
    
    # Mock NetworkX
    nx = MockNetworkX()
    nx.Graph = MockNetworkX.Graph
    nx.DiGraph = MockNetworkX.DiGraph
    nx.shortest_path = MockNetworkX.shortest_path
    nx.connected_components = MockNetworkX.connected_components
    sys.modules['networkx'] = nx
    
    # Mock aiosqlite
    aiosqlite = MockAiosqlite()
    aiosqlite.connect = MockAiosqlite.connect
    aiosqlite.Connection = MockAiosqlite.Connection
    aiosqlite.Cursor = MockAiosqlite.Cursor
    sys.modules['aiosqlite'] = aiosqlite
    
    # Mock FastAPI
    fastapi = MockFastAPI()
    fastapi.FastAPI = MockFastAPI.FastAPI
    fastapi.HTTPException = MockFastAPI.HTTPException
    fastapi.Request = MockFastAPI.Request
    fastapi.Depends = MockFastAPI.Depends
    fastapi.middleware = MockFastAPI.middleware
    fastapi.security = MockFastAPI.security
    sys.modules['fastapi'] = fastapi
    
    # Mock FastAPI middleware submodules
    middleware = MockFastAPI.middleware()
    cors = MockFastAPI.middleware.cors()
    cors.CORSMiddleware = MockFastAPI.middleware.cors.CORSMiddleware
    middleware.cors = cors
    sys.modules['fastapi.middleware'] = middleware
    sys.modules['fastapi.middleware.cors'] = cors
    
    # Mock FastAPI security submodules
    security = MockFastAPI.security()
    security.HTTPBearer = MockFastAPI.security.HTTPBearer
    security.HTTPAuthorizationCredentials = MockFastAPI.security.HTTPAuthorizationCredentials
    sys.modules['fastapi.security'] = security
    
    # Mock Uvicorn
    uvicorn = MockUvicorn()
    uvicorn.run = MockUvicorn.run
    sys.modules['uvicorn'] = uvicorn


# Auto-install when imported
install_mock_dependencies()