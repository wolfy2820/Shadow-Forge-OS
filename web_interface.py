#!/usr/bin/env python3
"""
ShadowForge AI OS Web Interface
Web-based UI for the AI Operating System with real-time monitoring
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any
import websockets
import threading
from pathlib import Path
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

class WebInterface:
    """Web interface for AI OS"""
    
    def __init__(self, ai_os, port=8080):
        self.ai_os = ai_os
        self.port = port
        self.logger = logging.getLogger("WebInterface")
        self.connected_clients = set()
        self.is_running = False
        
    async def start_server(self):
        """Start the web interface server"""
        try:
            self.logger.info(f"🌐 Starting web interface on port {self.port}")
            
            # Start WebSocket server for real-time communication
            start_server = websockets.serve(
                self.handle_websocket,
                "localhost",
                self.port + 1,  # WebSocket on port+1
                ping_interval=20,
                ping_timeout=10
            )
            
            # Start HTTP server for static files
            self._start_http_server()
            
            self.is_running = True
            
            await start_server
            
            self.logger.info(f"✅ Web interface running at http://localhost:{self.port}")
            
        except Exception as e:
            self.logger.error(f"❌ Web interface startup failed: {e}")
            raise
    
    def _start_http_server(self):
        """Start HTTP server for static files"""
        def run_http_server():
            try:
                # Create web directory
                web_dir = Path("/home/zeroday/ShadowForge-OS/web")
                web_dir.mkdir(exist_ok=True)
                
                # Create HTML interface
                self._create_web_files(web_dir)
                
                # Change to web directory
                import os
                os.chdir(web_dir)
                
                handler = http.server.SimpleHTTPRequestHandler
                with socketserver.TCPServer(("", self.port), handler) as httpd:
                    httpd.serve_forever()
                    
            except Exception as e:
                self.logger.error(f"HTTP server error: {e}")
        
        # Run HTTP server in separate thread
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()
    
    def _create_web_files(self, web_dir: Path):
        """Create web interface files"""
        
        # Create main HTML file
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShadowForge AI Operating System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.8);
            padding: 1rem 2rem;
            border-bottom: 2px solid #00ff88;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            color: #00ff88;
            font-size: 2rem;
            text-shadow: 0 0 10px #00ff88;
        }
        
        .header p {
            color: #cccccc;
            margin-top: 0.5rem;
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto;
            gap: 1rem;
            padding: 1rem;
            height: calc(100vh - 120px);
        }
        
        .panel {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid #333;
            border-radius: 10px;
            padding: 1rem;
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.1);
        }
        
        .panel h2 {
            color: #00ff88;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            border-bottom: 1px solid #333;
            padding-bottom: 0.5rem;
        }
        
        .terminal {
            grid-column: 1 / 3;
            background: #000;
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            height: 300px;
            overflow-y: auto;
        }
        
        .terminal-output {
            height: 200px;
            overflow-y: auto;
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #333;
            border-radius: 5px;
        }
        
        .terminal-input {
            display: flex;
            gap: 0.5rem;
        }
        
        .terminal-input input {
            flex: 1;
            background: #1a1a1a;
            border: 1px solid #00ff88;
            color: #00ff88;
            padding: 0.5rem;
            border-radius: 5px;
            font-family: inherit;
        }
        
        .terminal-input button {
            background: #00ff88;
            color: #000;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .terminal-input button:hover {
            background: #00cc6a;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        
        .status-item {
            background: rgba(0, 255, 136, 0.1);
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #333;
        }
        
        .status-item .label {
            color: #888;
            font-size: 0.8rem;
        }
        
        .status-item .value {
            color: #00ff88;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .business-list {
            max-height: 250px;
            overflow-y: auto;
        }
        
        .business-item {
            background: rgba(0, 255, 136, 0.1);
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 5px;
            border-left: 3px solid #00ff88;
        }
        
        .business-item .name {
            color: #00ff88;
            font-weight: bold;
        }
        
        .business-item .stats {
            color: #ccc;
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }
        
        .log-entry {
            color: #00ff88;
            margin-bottom: 0.3rem;
            font-size: 0.9rem;
        }
        
        .log-entry .timestamp {
            color: #888;
        }
        
        .log-entry .command {
            color: #fff;
        }
        
        .connection-status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .connected {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
            border: 1px solid #00ff88;
        }
        
        .disconnected {
            background: rgba(255, 0, 0, 0.2);
            color: #ff0000;
            border: 1px solid #ff0000;
        }
        
        .metric-card {
            background: linear-gradient(45deg, rgba(0, 255, 136, 0.1), rgba(0, 255, 136, 0.05));
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        .metric-label {
            color: #ccc;
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        
        @keyframes glow {
            0% { box-shadow: 0 0 5px rgba(0, 255, 136, 0.5); }
            50% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.8); }
            100% { box-shadow: 0 0 5px rgba(0, 255, 136, 0.5); }
        }
        
        .panel:hover {
            animation: glow 2s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 ShadowForge AI Operating System</h1>
        <p>The World's First AI-Controlled Business Operating System</p>
    </div>
    
    <div class="connection-status" id="connectionStatus">
        <span class="disconnected">🔴 Disconnected</span>
    </div>
    
    <div class="container">
        <!-- System Status Panel -->
        <div class="panel">
            <h2>📊 System Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">Uptime</div>
                    <div class="value" id="uptime">00:00:00</div>
                </div>
                <div class="status-item">
                    <div class="label">Commands</div>
                    <div class="value" id="commands">0</div>
                </div>
                <div class="status-item">
                    <div class="label">Wallet Balance</div>
                    <div class="value" id="walletBalance">$0.00</div>
                </div>
                <div class="status-item">
                    <div class="label">AI Intelligence</div>
                    <div class="value" id="aiIntelligence">1.0x</div>
                </div>
            </div>
            
            <div style="margin-top: 1rem;">
                <div class="metric-card">
                    <div class="metric-value" id="empireValuation">$0</div>
                    <div class="metric-label">Empire Valuation</div>
                </div>
            </div>
        </div>
        
        <!-- Business Portfolio Panel -->
        <div class="panel">
            <h2>🏢 Business Portfolio</h2>
            <div class="business-list" id="businessList">
                <div style="color: #888; text-align: center; padding: 2rem;">
                    No businesses created yet
                </div>
            </div>
        </div>
        
        <!-- Terminal Interface -->
        <div class="terminal">
            <h2 style="color: #00ff88; margin-bottom: 1rem;">🖥️ AI OS Terminal</h2>
            <div class="terminal-output" id="terminalOutput">
                <div class="log-entry">
                    <span class="timestamp">[System]</span> 
                    <span class="command">ShadowForge AI OS Ready. Type commands below...</span>
                </div>
            </div>
            <div class="terminal-input">
                <input type="text" id="commandInput" placeholder="Enter natural language command..." 
                       onkeypress="if(event.key==='Enter') sendCommand()">
                <button onclick="sendCommand()">Execute</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectInterval = null;
        
        function connectWebSocket() {
            const wsUrl = `ws://localhost:${parseInt(window.location.port) + 1}`;
            
            try {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('Connected to AI OS');
                    document.getElementById('connectionStatus').innerHTML = 
                        '<span class="connected">🟢 Connected</span>';
                    
                    if (reconnectInterval) {
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }
                    
                    // Request initial status
                    sendMessage({type: 'get_status'});
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    console.log('Disconnected from AI OS');
                    document.getElementById('connectionStatus').innerHTML = 
                        '<span class="disconnected">🔴 Disconnected</span>';
                    
                    // Attempt to reconnect every 5 seconds
                    if (!reconnectInterval) {
                        reconnectInterval = setInterval(connectWebSocket, 5000);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
                
            } catch (error) {
                console.error('Failed to connect:', error);
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                }
            }
        }
        
        function sendMessage(message) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(message));
            }
        }
        
        function sendCommand() {
            const input = document.getElementById('commandInput');
            const command = input.value.trim();
            
            if (command) {
                // Add to terminal output
                addTerminalEntry(`[User] ${command}`, 'command');
                
                // Send to AI OS
                sendMessage({
                    type: 'command',
                    command: command
                });
                
                input.value = '';
            }
        }
        
        function addTerminalEntry(text, type = 'log') {
            const output = document.getElementById('terminalOutput');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            const timestamp = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> <span class="${type}">${text}</span>`;
            
            output.appendChild(entry);
            output.scrollTop = output.scrollHeight;
            
            // Keep only last 100 entries
            while (output.children.length > 100) {
                output.removeChild(output.firstChild);
            }
        }
        
        function handleMessage(data) {
            switch (data.type) {
                case 'status':
                    updateStatus(data.status);
                    break;
                    
                case 'command_result':
                    handleCommandResult(data.result);
                    break;
                    
                case 'business_update':
                    updateBusinessList(data.businesses);
                    break;
                    
                case 'system_notification':
                    addTerminalEntry(`[System] ${data.message}`, 'log');
                    break;
                    
                default:
                    console.log('Unknown message type:', data);
            }
        }
        
        function updateStatus(status) {
            document.getElementById('uptime').textContent = status.uptime || '00:00:00';
            document.getElementById('commands').textContent = status.session_stats?.commands_processed || 0;
            document.getElementById('walletBalance').textContent = 
                `$${(status.developer_panel?.system_health?.wallet_balance || 0).toFixed(2)}`;
            document.getElementById('empireValuation').textContent = 
                `$${(status.session_stats?.total_business_revenue || 0).toLocaleString()}`;
        }
        
        function handleCommandResult(result) {
            if (result.success) {
                addTerminalEntry(`[AI] ✅ Command executed successfully`, 'log');
                
                // Handle specific result types
                if (result.businesses) {
                    updateBusinessList(result.businesses);
                } else if (result.balance !== undefined) {
                    addTerminalEntry(`[AI] 💰 Wallet Balance: $${result.balance.toFixed(2)}`, 'log');
                } else if (result.output) {
                    addTerminalEntry(`[AI] ${result.output}`, 'log');
                }
                
                // Request updated status
                sendMessage({type: 'get_status'});
                
            } else {
                addTerminalEntry(`[AI] ❌ Error: ${result.error}`, 'error');
                if (result.suggestion) {
                    addTerminalEntry(`[AI] 💡 ${result.suggestion}`, 'log');
                }
            }
        }
        
        function updateBusinessList(businesses) {
            const list = document.getElementById('businessList');
            
            if (!businesses || businesses.length === 0) {
                list.innerHTML = '<div style="color: #888; text-align: center; padding: 2rem;">No businesses created yet</div>';
                return;
            }
            
            list.innerHTML = businesses.map(business => `
                <div class="business-item">
                    <div class="name">${business.name}</div>
                    <div class="stats">
                        Status: ${business.status} | 
                        Revenue: $${(business.total_revenue || 0).toFixed(2)} |
                        Progress: ${(business.development_progress || 0).toFixed(1)}%
                    </div>
                </div>
            `).join('');
        }
        
        // Initialize connection when page loads
        window.addEventListener('load', function() {
            connectWebSocket();
            
            // Request status updates every 10 seconds
            setInterval(function() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    sendMessage({type: 'get_status'});
                }
            }, 10000);
        });
        
        // Handle enter key in command input
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('commandInput').focus();
        });
    </script>
</body>
</html>"""
        
        with open(web_dir / "index.html", "w") as f:
            f.write(html_content)
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        try:
            self.connected_clients.add(websocket)
            self.logger.info(f"🔌 Client connected: {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.handle_websocket_message(data)
                    
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error", 
                        "message": str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            self.logger.info("🔌 Client disconnected")
    
    async def handle_websocket_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebSocket message from client"""
        try:
            message_type = data.get("type")
            
            if message_type == "get_status":
                # Return system status
                status = self.ai_os.get_system_status()
                return {
                    "type": "status",
                    "status": status
                }
            
            elif message_type == "command":
                # Execute command through AI OS
                command = data.get("command", "")
                result = await self.ai_os.execute_command(command)
                
                return {
                    "type": "command_result",
                    "result": result
                }
            
            elif message_type == "get_businesses":
                # Return business list
                return {
                    "type": "business_update",
                    "businesses": self.ai_os.business.active_businesses
                }
            
            else:
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            return {
                "type": "error",
                "message": str(e)
            }
    
    async def broadcast_update(self, update_data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if self.connected_clients:
            message = json.dumps(update_data)
            
            # Send to all connected clients
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except:
                    disconnected.add(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.connected_clients.discard(client)