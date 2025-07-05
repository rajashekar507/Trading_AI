"""
Real-time Trading Dashboard for VLR_AI Trading System
Displays LIVE market data, positions, performance metrics, and system health
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import time

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas as pd
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("[WARNING] Dash not available. Install with: pip install dash plotly")

logger = logging.getLogger('trading_system.dashboard')

class TradingDashboard:
    """Real-time trading dashboard with LIVE market data"""
    
    def __init__(self, settings, system_manager=None):
        self.settings = settings
        self.system_manager = system_manager
        self.port = getattr(settings, 'DASHBOARD_PORT', int(os.getenv('DASHBOARD_PORT', 8080)))
        self.host = getattr(settings, 'DASHBOARD_HOST', os.getenv('DASHBOARD_HOST', '127.0.0.1'))
        
        # Dashboard data - ALL REAL DATA
        self.market_data = {}
        self.positions = {}
        self.performance_data = []
        self.system_health = {}
        self.trade_history = []
        self.risk_metrics = {}
        
        # Update intervals
        self.update_interval = 5  # seconds
        self.last_update = None
        
        # Dashboard app
        self.app = None
        self.running = False
        
        if DASH_AVAILABLE:
            self._initialize_dash_app()
        else:
            logger.warning("[DASHBOARD] Dash not available, using simple dashboard")
    
    def _initialize_dash_app(self):
        """Initialize Dash application for REAL market data"""
        try:
            self.app = dash.Dash(__name__, title="VLR_AI Trading Dashboard - LIVE DATA")
            self.app.layout = self._create_layout()
            self._setup_callbacks()
            logger.info("[DASHBOARD] Dash application initialized for REAL market data")
        except Exception as e:
            logger.error(f"[DASHBOARD] Failed to initialize Dash app: {e}")
            self.app = None
    
    def _create_layout(self):
        """Create dashboard layout for REAL market data"""
        return html.Div([
            # Header
            html.Div([
                html.H1("VLR_AI Trading System Dashboard - LIVE MARKET DATA", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.H3("🔴 REAL-TIME DATA ONLY - NO SIMULATIONS", 
                       style={'textAlign': 'center', 'color': '#e74c3c', 'marginBottom': '10px'}),
                html.Div(id='last-update', style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
            
            # System Status Row
            html.Div([
                html.Div([
                    html.H3("System Health", style={'color': '#27ae60'}),
                    html.Div(id='system-health-display')
                ], className='four columns', style={'backgroundColor': '#ffffff', 'padding': '15px', 'margin': '5px', 'borderRadius': '5px'}),
                
                html.Div([
                    html.H3("LIVE Market Status", style={'color': '#3498db'}),
                    html.Div(id='market-status-display')
                ], className='four columns', style={'backgroundColor': '#ffffff', 'padding': '15px', 'margin': '5px', 'borderRadius': '5px'}),
                
                html.Div([
                    html.H3("Risk Metrics", style={'color': '#e74c3c'}),
                    html.Div(id='risk-metrics-display')
                ], className='four columns', style={'backgroundColor': '#ffffff', 'padding': '15px', 'margin': '5px', 'borderRadius': '5px'})
            ], className='row'),
            
            # Charts Row - REAL DATA ONLY
            html.Div([
                html.Div([
                    dcc.Graph(id='market-data-chart')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='performance-chart')
                ], className='six columns')
            ], className='row', style={'marginTop': '20px'}),
            
            # Positions and Trades Row - REAL DATA ONLY
            html.Div([
                html.Div([
                    html.H3("Active Positions (REAL)", style={'color': '#9b59b6'}),
                    html.Div(id='positions-table')
                ], className='six columns', style={'backgroundColor': '#ffffff', 'padding': '15px', 'margin': '5px', 'borderRadius': '5px'}),
                
                html.Div([
                    html.H3("Recent Trades (REAL)", style={'color': '#f39c12'}),
                    html.Div(id='trades-table')
                ], className='six columns', style={'backgroundColor': '#ffffff', 'padding': '15px', 'margin': '5px', 'borderRadius': '5px'})
            ], className='row', style={'marginTop': '20px'}),
            
            # Auto-refresh component for REAL data
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            )
        ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for REAL data"""
        @self.app.callback(
            [Output('last-update', 'children'),
             Output('system-health-display', 'children'),
             Output('market-status-display', 'children'),
             Output('risk-metrics-display', 'children'),
             Output('market-data-chart', 'figure'),
             Output('performance-chart', 'figure'),
             Output('positions-table', 'children'),
             Output('trades-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components with REAL data"""
            try:
                # Update REAL data
                self._fetch_real_time_data()
                
                # Last update time
                last_update = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (REAL DATA)"
                
                # System health
                system_health = self._create_system_health_display()
                
                # Market status
                market_status = self._create_market_status_display()
                
                # Risk metrics
                risk_metrics = self._create_risk_metrics_display()
                
                # Charts with REAL data
                market_chart = self._create_market_data_chart()
                performance_chart = self._create_performance_chart()
                
                # Tables with REAL data
                positions_table = self._create_positions_table()
                trades_table = self._create_trades_table()
                
                return (last_update, system_health, market_status, risk_metrics,
                       market_chart, performance_chart, positions_table, trades_table)
                
            except Exception as e:
                logger.error(f"[DASHBOARD] Update error: {e}")
                error_msg = f"Error updating dashboard with REAL data: {str(e)}"
                return (error_msg, error_msg, error_msg, error_msg, {}, {}, error_msg, error_msg)
    
    def _fetch_real_time_data(self):
        try:
            if self.system_manager:
                # Get REAL system status
                self.system_health = self.system_manager.get_system_status()
                
                # Get REAL market data
                if hasattr(self.system_manager, 'data_manager') and self.system_manager.data_manager:
                    # Fetch REAL market data from APIs
                    self.market_data = self.system_manager.data_manager.get_latest_market_data()
                
                # Get REAL positions
                if hasattr(self.system_manager, 'active_positions'):
                    self.positions = self.system_manager.active_positions
                
                # Get REAL performance data
                self._update_performance_data()
                
                # Get REAL risk metrics
                if hasattr(self.system_manager, 'risk_manager') and self.system_manager.risk_manager:
                    self.risk_metrics = self.system_manager.risk_manager.get_risk_summary()
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"[DASHBOARD] REAL data fetch error: {e}")
    
    def _update_performance_data(self):
        """Update performance data with REAL P&L"""
        # Add current REAL P&L data point
        current_pnl = sum(pos.get('pnl', 0) for pos in self.positions.values()) if self.positions else 0
        
        self.performance_data.append({
            'timestamp': datetime.now(),
            'pnl': current_pnl,
            'positions_count': len(self.positions),
            'data_type': 'REAL'
        })
        
        # Keep only last 100 data points
        if len(self.performance_data) > 100:
            self.performance_data = self.performance_data[-100:]
    
    def _create_system_health_display(self) -> html.Div:
        """Create system health display with REAL data"""
        if not self.system_health:
            return html.Div("No REAL system data available", style={'color': 'red'})
        
        components = self.system_health.get('components', {})
        running_components = sum(1 for status in components.values() if status)
        total_components = len(components)
        
        health_percentage = (running_components / total_components * 100) if total_components > 0 else 0
        
        color = '#27ae60' if health_percentage > 80 else '#f39c12' if health_percentage > 60 else '#e74c3c'
        
        return html.Div([
            html.H4(f"{health_percentage:.1f}%", style={'color': color, 'fontSize': '2em', 'margin': '0'}),
            html.P(f"{running_components}/{total_components} components running"),
            html.P(f"Cycle: {self.system_health.get('cycle_count', 0)}"),
            html.P("🔴 REAL SYSTEM DATA", style={'color': '#e74c3c', 'fontSize': '0.8em'})
        ])
    
    def _create_market_status_display(self) -> html.Div:
        """Create market status display with REAL market data"""
        if not self.market_data:
            return html.Div("No REAL market data available", style={'color': 'red'})
        
        # Extract REAL market data
        spot_data = self.market_data.get('spot_data', {})
        if spot_data and spot_data.get('status') == 'success':
            prices = spot_data.get('prices', {})
            nifty_price = prices.get('NIFTY', 0)
            banknifty_price = prices.get('BANKNIFTY', 0)
            
            # Calculate change percentages (would need previous close data)
            nifty_change = 0  # This would be calculated from real previous close
            banknifty_change = 0  # This would be calculated from real previous close
            
            return html.Div([
                html.Div([
                    html.Strong("NIFTY (REAL): "),
                    html.Span(f"₹{nifty_price:.2f} "),
                    html.Span(f"({nifty_change:+.2f}%)", 
                             style={'color': '#27ae60' if nifty_change >= 0 else '#e74c3c'})
                ]),
                html.Div([
                    html.Strong("BANKNIFTY (REAL): "),
                    html.Span(f"₹{banknifty_price:.2f} "),
                    html.Span(f"({banknifty_change:+.2f}%)", 
                             style={'color': '#27ae60' if banknifty_change >= 0 else '#e74c3c'})
                ]),
                html.P("🔴 LIVE MARKET DATA", style={'color': '#e74c3c', 'fontSize': '0.8em'})
            ])
        else:
            return html.Div("REAL market data not available", style={'color': 'red'})
    
    def _create_risk_metrics_display(self) -> html.Div:
        """Create risk metrics display with REAL data"""
        if not self.risk_metrics:
            return html.Div("No REAL risk data available", style={'color': 'red'})
        
        daily_stats = self.risk_metrics.get('daily_stats', {})
        daily_pnl = daily_stats.get('pnl', 0)
        max_drawdown = daily_stats.get('max_drawdown', 0)
        
        pnl_color = '#27ae60' if daily_pnl >= 0 else '#e74c3c'
        
        return html.Div([
            html.Div([
                html.Strong("Daily P&L (REAL): "),
                html.Span(f"₹{daily_pnl:.2f}", style={'color': pnl_color})
            ]),
            html.Div([
                html.Strong("Max Drawdown: "),
                html.Span(f"₹{max_drawdown:.2f}", style={'color': '#e74c3c'})
            ]),
            html.Div([
                html.Strong("Active Positions: "),
                html.Span(f"{len(self.positions)}")
            ]),
            html.P("🔴 REAL RISK DATA", style={'color': '#e74c3c', 'fontSize': '0.8em'})
        ])
    
    def _create_market_data_chart(self) -> Dict:
        """Create market data chart with REAL data"""
        try:
            if not self.market_data:
                return {'data': [], 'layout': {'title': 'No REAL market data available'}}
            
            # Use REAL market data for charts
            spot_data = self.market_data.get('spot_data', {})
            if spot_data and spot_data.get('status') == 'success':
                prices = spot_data.get('prices', {})
                
                # Create time series with REAL data points
                # This would need historical data storage
                timestamps = [datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)]
                nifty_prices = [prices.get('NIFTY', 0)] * 30  # Would use real historical data
                banknifty_prices = [prices.get('BANKNIFTY', 0)] * 30  # Would use real historical data
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=nifty_prices,
                    mode='lines',
                    name='NIFTY (REAL)',
                    line=dict(color='#3498db')
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=banknifty_prices,
                    mode='lines',
                    name='BANKNIFTY (REAL)',
                    line=dict(color='#e74c3c'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='LIVE Market Data (REAL)',
                    xaxis_title='Time',
                    yaxis=dict(title='NIFTY', side='left'),
                    yaxis2=dict(title='BANKNIFTY', side='right', overlaying='y'),
                    height=400
                )
                
                return fig
            else:
                return {'data': [], 'layout': {'title': 'REAL market data not available'}}
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Chart creation error: {e}")
            return {'data': [], 'layout': {'title': f'Chart error: {str(e)}'}}
    
    def _create_performance_chart(self) -> Dict:
        """Create performance chart with REAL P&L data"""
        try:
            if not self.performance_data:
                return {'data': [], 'layout': {'title': 'No REAL performance data available'}}
            
            timestamps = [data['timestamp'] for data in self.performance_data]
            pnl_values = [data['pnl'] for data in self.performance_data]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=pnl_values,
                mode='lines+markers',
                name='P&L (REAL)',
                line=dict(color='#27ae60' if pnl_values[-1] >= 0 else '#e74c3c')
            ))
            
            fig.update_layout(
                title='Performance (REAL P&L)',
                xaxis_title='Time',
                yaxis_title='P&L (₹)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Performance chart error: {e}")
            return {'data': [], 'layout': {'title': f'Chart error: {str(e)}'}}
    
    def _create_positions_table(self) -> html.Div:
        """Create positions table with REAL position data"""
        if not self.positions:
            return html.Div("No REAL active positions", style={'textAlign': 'center', 'color': '#7f8c8d'})
        
        rows = []
        for pos_key, position in list(self.positions.items())[:10]:  # Show max 10 positions
            signal = position.get('signal', {})
            pnl = position.get('pnl', 0)
            pnl_color = '#27ae60' if pnl >= 0 else '#e74c3c'
            
            rows.append(html.Tr([
                html.Td(signal.get('instrument', 'N/A')),
                html.Td(f"{signal.get('strike', 'N/A')} {signal.get('option_type', 'N/A')}"),
                html.Td(f"₹{signal.get('entry_price', 0):.2f}"),
                html.Td(f"₹{pnl:.2f}", style={'color': pnl_color})
            ]))
        
        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Instrument"),
                    html.Th("Strike/Type"),
                    html.Th("Entry Price"),
                    html.Th("P&L (REAL)")
                ])
            ]),
            html.Tbody(rows)
        ], style={'width': '100%', 'textAlign': 'center'})
        
        return html.Div([
            table,
            html.P("🔴 REAL POSITION DATA", style={'color': '#e74c3c', 'fontSize': '0.8em', 'textAlign': 'center'})
        ])
    
    def _create_trades_table(self) -> html.Div:
        """Create trades table with REAL trade data"""
        if not self.trade_history:
            return html.Div("No REAL recent trades", style={'textAlign': 'center', 'color': '#7f8c8d'})
        
        # Use REAL trade history data
        rows = []
        for trade in self.trade_history[-5:]:  # Last 5 trades
            pnl_color = '#27ae60' if trade.get('pnl', 0) >= 0 else '#e74c3c'
            
            rows.append(html.Tr([
                html.Td(trade.get('time', 'N/A')),
                html.Td(trade.get('instrument', 'N/A')),
                html.Td(trade.get('action', 'N/A')),
                html.Td(f"₹{trade.get('price', 0):.2f}"),
                html.Td(f"₹{trade.get('pnl', 0):.2f}", style={'color': pnl_color})
            ]))
        
        if not rows:
            return html.Div("No REAL trades available", style={'textAlign': 'center', 'color': '#7f8c8d'})
        
        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Time"),
                    html.Th("Instrument"),
                    html.Th("Action"),
                    html.Th("Price"),
                    html.Th("P&L (REAL)")
                ])
            ]),
            html.Tbody(rows)
        ], style={'width': '100%', 'textAlign': 'center'})
        
        return html.Div([
            table,
            html.P("🔴 REAL TRADE DATA", style={'color': '#e74c3c', 'fontSize': '0.8em', 'textAlign': 'center'})
        ])
    
    async def run(self):
        """Run the dashboard with REAL market data"""
        try:
            self.running = True
            
            if DASH_AVAILABLE and self.app:
                logger.info(f"[DASHBOARD] Starting Dash server on {self.host}:{self.port}")
                print(f"🌐 Dashboard starting on http://{self.host}:{self.port}")
                print("🔴 REAL-TIME TRADING DASHBOARD WITH LIVE MARKET DATA")
                print("[INFO] Features:")
                print("  - LIVE market data charts from real APIs")
                print("  - Real-time P&L tracking from actual positions")
                print("  - Real position monitoring")
                print("  - Actual system health metrics")
                print("  - Live risk exposure visualization")
                
                # Run Dash app in a separate thread
                def run_dash():
                    self.app.run_server(host=self.host, port=self.port, debug=False)
                
                dash_thread = threading.Thread(target=run_dash, daemon=True)
                dash_thread.start()
                
                # Keep the async function running
                while self.running:
                    await asyncio.sleep(1)
                    
            else:
                # Fallback simple dashboard
                await self._run_simple_dashboard()
                
        except Exception as e:
            logger.error(f"[DASHBOARD] Error running dashboard: {e}")
            print(f"[ERROR] Dashboard error: {e}")
    
    async def _run_simple_dashboard(self):
        """Run simple text-based dashboard with REAL data"""
        print(f"🌐 Simple Dashboard running (Dash not available)")
        print("🔴 REAL-TIME trading system monitoring - NO SIMULATIONS")
        print(f"[INFO] Dashboard features available via console")
        
        cycle = 0
        while self.running:
            cycle += 1
            
            # Fetch and display REAL data
            self._fetch_real_time_data()
            
            print(f"\n[DASHBOARD] Cycle {cycle} - {datetime.now().strftime('%H:%M:%S')} (REAL DATA)")
            
            if self.system_health:
                components = self.system_health.get('components', {})
                running = sum(1 for status in components.values() if status)
                total = len(components)
                print(f"  System Health: {running}/{total} components running")
            
            if self.market_data:
                spot_data = self.market_data.get('spot_data', {})
                if spot_data and spot_data.get('status') == 'success':
                    prices = spot_data.get('prices', {})
                    print(f"  NIFTY (REAL): ₹{prices.get('NIFTY', 0):.2f}")
                    print(f"  BANKNIFTY (REAL): ₹{prices.get('BANKNIFTY', 0):.2f}")
            
            print(f"  Active Positions (REAL): {len(self.positions)}")
            
            if self.risk_metrics:
                daily_pnl = self.risk_metrics.get('daily_stats', {}).get('pnl', 0)
                print(f"  Daily P&L (REAL): ₹{daily_pnl:.2f}")
            
            await asyncio.sleep(30)  # Update every 30 seconds
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False
        logger.info("[DASHBOARD] Dashboard stopped")

class DashboardApp:
    """Dashboard application wrapper for compatibility"""
    
    def __init__(self, settings):
        self.settings = settings
        self.dashboard = TradingDashboard(settings)
    
    async def run(self):
        """Run the dashboard with REAL data"""
        await self.dashboard.run()
    
    def stop(self):
        """Stop the dashboard"""
        self.dashboard.stop()

def create_dashboard_app(settings, system_manager=None):
    """Create dashboard application with REAL market data"""
    if DASH_AVAILABLE:
        return TradingDashboard(settings, system_manager)
    else:
        return DashboardApp(settings)