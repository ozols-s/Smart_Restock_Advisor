import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
from datetime import datetime
from scipy import stats
import holidays
from decimal import Decimal
from sqlalchemy import text
from flask import jsonify, request
import json

class RecommendedOrder:
    def convert_decimals_to_floats(df):
        if df.empty:
            return df
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna()
                if not sample.empty and isinstance(sample.iloc[0], Decimal):
                    df[col] = df[col].apply(lambda x: float(x) if x is not None else x)
            elif df[col].dtype.name == 'decimal':
                df[col] = df[col].astype(float)
        return df

    def convert_dataframes_to_float(self, forecast_df, current_stock_df, orders_df):
        forecast_df = self._convert_df_to_float(forecast_df)
        current_stock_df = self._convert_df_to_float(current_stock_df)
        if not orders_df.empty:
            orders_df = self._convert_df_to_float(orders_df)
        return forecast_df, current_stock_df, orders_df

    def _convert_df_to_float(self, df):
        if df.empty:
            return df
        df = df.copy()
        for col in df.columns:
            if col in ['Date', 'date', 'SKU', 'sku', 'product_code', 'status', 'expected_delivery']:
                continue
            try:
                original_dtype = df[col].dtype
                original_sample = df[col].head(3).tolist() if len(df) > 0 else []
                df[col] = df[col].astype(float)
                result_sample = df[col].head(3).tolist() if len(df) > 0 else []
            except Exception as e:
                try:
                    df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) else x)
                except:
                    pass
        return df

    def calculate_recommended_order(self, forecast_df, current_stock_df, orders_df, business_params):
        lead_time = business_params['lead_time']
        min_batch = business_params['min_batch']
        safety_stock = business_params['safety_stock']
        reorder_point = business_params.get('reorder_point')
        forecast_df = forecast_df.copy()
        current_stock_df = current_stock_df.copy()
        orders_df = orders_df.copy()
        forecast_df = RecommendedOrder.convert_decimals_to_floats(forecast_df)
        current_stock_df = RecommendedOrder.convert_decimals_to_floats(current_stock_df)
        if not orders_df.empty:
            orders_df = RecommendedOrder.convert_decimals_to_floats(orders_df)
        forecast_df, current_stock_df, orders_df = self.convert_dataframes_to_float(
            forecast_df, current_stock_df, orders_df
        )
        if 'Date' in forecast_df.columns:
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
            forecast_df = forecast_df.dropna(subset=['Date'])

        def standardize_sku_column(df, df_name="df"):
            if 'SKU' in df.columns:
                pass
            else:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'sku' in col_lower or 'product' in col_lower or 'code' in col_lower:
                        df['SKU'] = df[col]
                        break
                else:
                    if len(df.columns) > 0:
                        df['SKU'] = df[df.columns[0]]
                    else:
                        raise ValueError(f"Не найдена колонка с SKU в {df_name}")
            df['SKU'] = df['SKU'].astype(str).str.strip()
            return df

        forecast_df = standardize_sku_column(forecast_df, "forecast_df")
        current_stock_df = standardize_sku_column(current_stock_df, "current_stock_df")
        if not orders_df.empty:
            try:
                orders_df = standardize_sku_column(orders_df, "orders_df")
            except:
                orders_df['SKU'] = ''
        stock_col_candidates = ['stock', 'value', 'quantity', 'qty', 'остаток', 'кол-во']
        stock_value_col = None
        for col in current_stock_df.columns:
            col_lower = str(col).lower()
            if any(candidate in col_lower for candidate in stock_col_candidates):
                stock_value_col = col
                break
        if stock_value_col is None:
            numeric_cols = current_stock_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stock_value_col = numeric_cols[0]
            else:
                if len(current_stock_df.columns) > 1:
                    stock_value_col = current_stock_df.columns[1]
        forecast_skus = set(forecast_df['SKU'].unique())
        stock_skus = set(current_stock_df['SKU'].unique())
        common_skus = forecast_skus.intersection(stock_skus)
        if not common_skus:
            return pd.DataFrame()
        sku_results = []
        for sku in common_skus:
            sku_stock = current_stock_df[current_stock_df['SKU'] == sku]
            if not sku_stock.empty:
                try:
                    stock_value = sku_stock[stock_value_col].values[0]
                    if isinstance(stock_value, (Decimal, np.number)):
                        current_stock = float(stock_value)
                    else:
                        current_stock = float(stock_value)
                except Exception as e:
                    current_stock = 0.0
            else:
                current_stock = 0.0
            sku_forecast = forecast_df[forecast_df['SKU'] == sku].copy()
            if sku_forecast.empty:
                continue
            if not orders_df.empty and 'SKU' in orders_df.columns:
                sku_orders = orders_df[orders_df['SKU'] == sku].copy()
            else:
                sku_orders = pd.DataFrame()
            stocks_df = self.calculate_stocks(
                current_stock=current_stock,
                forecast_df=sku_forecast,
                orders_df=sku_orders
            )
            demand_during_lead_time = 0.0
            if 'Date' in sku_forecast.columns and not sku_forecast.empty:
                sku_forecast_sorted = sku_forecast.sort_values('Date')
                if len(sku_forecast_sorted) >= lead_time:
                    lead_time_forecast = sku_forecast_sorted.head(lead_time)
                else:
                    lead_time_forecast = sku_forecast_sorted
                demand_during_lead_time = float(lead_time_forecast['Value'].sum())
            else:
                demand_during_lead_time = float(sku_forecast['Value'].sum()) if not sku_forecast.empty else 0.0
            min_projected_stock = current_stock
            if not stocks_df.empty and 'Date' in stocks_df.columns and 'Stock' in stocks_df.columns:
                stocks_df_sorted = stocks_df.sort_values('Date')
                if len(stocks_df_sorted) >= lead_time:
                    lead_time_stocks = stocks_df_sorted.head(lead_time)
                    min_projected_stock = float(lead_time_stocks['Stock'].min())
                else:
                    min_projected_stock = float(stocks_df_sorted['Stock'].min())
            orders_in_transit = 0.0
            if not sku_orders.empty and 'expected_delivery' in sku_orders.columns and 'quantity' in sku_orders.columns:
                sku_orders_processed = sku_orders.copy()
                sku_orders_processed['expected_delivery'] = pd.to_datetime(
                    sku_orders_processed['expected_delivery'], errors='coerce'
                )
                sku_orders_processed = sku_orders_processed.dropna(subset=['expected_delivery'])
                relevant_statuses = ['processing', 'pending', 'shipped', 'in_transit']
                if 'status' in sku_orders_processed.columns:
                    transit_orders = sku_orders_processed[
                        sku_orders_processed['status'].isin(relevant_statuses)
                    ]
                else:
                    transit_orders = sku_orders_processed
                orders_in_transit = float(transit_orders['quantity'].sum()) if not transit_orders.empty else 0.0
            required_stock = float(demand_during_lead_time) + float(safety_stock)
            available_stock = float(current_stock) + float(orders_in_transit)
            recommended_qty = max(0.0, required_stock - available_stock)
            if recommended_qty > 0 and recommended_qty < min_batch:
                recommended_qty = float(min_batch)
            status, priority = self.determine_order_status(
                current_stock, safety_stock, reorder_point, recommended_qty
            )
            stockout_date = self.calculate_stockout_date_from_stocks(
                stocks_df, safety_stock
            )
            valid_statuses = ["КРИТИЧЕСКИЙ ДЕФИЦИТ", "НИЖЕ ТОЧКИ ЗАКАЗА", "ПЛАНОВЫЙ ЗАКАЗ"]
            if status in valid_statuses:
                result = {
                    'SKU': sku,
                    'recommended_quantity': float(round(recommended_qty, 2)),
                    'status': status,
                    'priority': priority,
                    'current_stock': float(current_stock),
                    'orders_in_transit': float(orders_in_transit),
                    'min_projected_stock': float(round(min_projected_stock, 2)),
                    'demand_during_lead_time': float(round(demand_during_lead_time, 2)),
                    'safety_stock': float(safety_stock),
                    'lead_time_days': int(lead_time),
                    'min_batch_size': int(min_batch),
                    'expected_stockout_date': stockout_date,
                    'reorder_point': float(reorder_point) if reorder_point else None,
                    'reason': self.generate_reason_with_transit(
                        status, current_stock, safety_stock, recommended_qty, orders_in_transit
                    )
                }
                sku_results.append(result)
        return pd.DataFrame(sku_results) if sku_results else pd.DataFrame()

    def calculate_stocks(self, current_stock, forecast_df, orders_df):
        stocks_df = forecast_df.copy()
        if 'Date' not in stocks_df.columns or stocks_df.empty:
            return pd.DataFrame({
                'Date': [pd.Timestamp.now().normalize()],
                'Stock': [float(current_stock)]
            })
        if 'Date' in stocks_df.columns:
            stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], errors='coerce')
            stocks_df = stocks_df.dropna(subset=['Date'])
            stocks_df = stocks_df.sort_values('Date')
        if stocks_df.empty:
            return pd.DataFrame({
                'Date': [pd.Timestamp.now().normalize()],
                'Stock': [float(current_stock)]
            })
        if 'Value' in stocks_df.columns:
            stocks_df['Value'] = stocks_df['Value'].astype(float)
        stocks_df['Cumulative_Forecast'] = stocks_df['Value'].cumsum()
        stocks_df['Stock'] = float(current_stock) - stocks_df['Cumulative_Forecast']
        if not orders_df.empty and 'expected_delivery' in orders_df.columns and 'quantity' in orders_df.columns:
            orders_processed = orders_df.copy()
            if 'quantity' in orders_processed.columns:
                orders_processed['quantity'] = orders_processed['quantity'].astype(float)
            relevant_statuses = ['processing', 'pending', 'shipped', 'in_transit', 'confirmed']
            if 'status' in orders_processed.columns:
                orders_processed = orders_processed[
                    orders_processed['status'].isin(relevant_statuses)
                ]
            if not orders_processed.empty:
                orders_processed['expected_delivery'] = pd.to_datetime(
                    orders_processed['expected_delivery'], errors='coerce'
                )
                orders_processed = orders_processed.dropna(subset=['expected_delivery'])
                if not orders_processed.empty:
                    deliveries = orders_processed.groupby('expected_delivery').agg({
                        'quantity': 'sum'
                    }).reset_index()
                    deliveries.columns = ['Date', 'Delivery_Quantity']
                    deliveries = deliveries.sort_values('Date')
                    deliveries['Delivery_Quantity'] = deliveries['Delivery_Quantity'].astype(float)
                    deliveries['Cumulative_Delivery'] = deliveries['Delivery_Quantity'].cumsum()
                    stocks_df = pd.merge_asof(
                        stocks_df[['Date', 'Stock', 'Cumulative_Forecast']],
                        deliveries[['Date', 'Cumulative_Delivery']],
                        on='Date',
                        direction='backward'
                    )
                    stocks_df['Cumulative_Delivery'] = stocks_df['Cumulative_Delivery'].fillna(0.0)
                    stocks_df['Stock'] = stocks_df['Stock'].astype(float)
                    stocks_df['Cumulative_Delivery'] = stocks_df['Cumulative_Delivery'].astype(float)
                    stocks_df['Stock'] = stocks_df['Stock'] + stocks_df['Cumulative_Delivery']
        result_df = stocks_df[['Date', 'Stock']].copy()
        result_df['Stock'] = result_df['Stock'].astype(float)
        if 'SKU' in forecast_df.columns and not forecast_df.empty:
            sku_value = forecast_df['SKU'].iloc[0]
            result_df['SKU'] = sku_value
        return result_df

    def calculate_stockout_date_from_stocks(self, stocks_df, safety_stock):
        if stocks_df.empty or 'Date' not in stocks_df.columns or 'Stock' not in stocks_df.columns:
            return "Нет данных"
        stocks_df = stocks_df.sort_values('Date')
        safety_stock = float(safety_stock)
        for idx, row in stocks_df.iterrows():
            stock_value = float(row['Stock']) if pd.notnull(row['Stock']) else float('inf')
            if stock_value <= safety_stock:
                try:
                    if pd.api.types.is_datetime64_any_dtype(row['Date']):
                        return row['Date'].strftime('%Y-%m-%d')
                    else:
                        return str(row['Date'])
                except:
                    return str(row['Date'])
        return "Более чем через 14 дней"

    def determine_order_status(self, current_stock, safety_stock, reorder_point, recommended_qty):
        current_stock = float(current_stock)
        safety_stock = float(safety_stock)
        recommended_qty = float(recommended_qty)
        reorder_point = float(reorder_point) if reorder_point else None
        if recommended_qty <= 0:
            if current_stock > (safety_stock * 3):
                return "ИЗБЫТОК - НЕ ЗАКАЗЫВАТЬ", "LOW"
            else:
                return "НЕ ТРЕБУЕТСЯ", "LOW"
        if current_stock <= safety_stock:
            return "КРИТИЧЕСКИЙ ДЕФИЦИТ", "HIGH"
        if reorder_point and current_stock <= reorder_point:
            return "НИЖЕ ТОЧКИ ЗАКАЗА", "MEDIUM"
        return "ПЛАНОВЫЙ ЗАКАЗ", "NORMAL"

    def generate_reason_with_transit(self, status, current_stock, safety_stock, recommended_qty, orders_in_transit):
        current_stock = float(current_stock)
        safety_stock = float(safety_stock)
        recommended_qty = float(recommended_qty)
        orders_in_transit = float(orders_in_transit)
        reasons = {
            "КРИТИЧЕСКИЙ ДЕФИЦИТ":
                f"Текущий остаток {current_stock:.1f} шт. ниже страхового запаса {safety_stock:.1f} шт. "
                f"(Заказов в пути: {orders_in_transit:.1f} шт.)",
            "НИЖЕ ТОЧКИ ЗАКАЗА":
                f"Остаток {current_stock:.1f} шт. достиг точки заказа. "
                f"(Заказов в пути: {orders_in_transit:.1f} шт.)",
            "ПЛАНОВЫЙ ЗАКАЗ":
                f"Плановое пополнение для поддержания запасов. "
                f"(Текущий запас: {current_stock:.1f} шт., в пути: {orders_in_transit:.1f} шт.)",
            "НЕ ТРЕБУЕТСЯ":
                f"Достаточный запас на складе ({current_stock:.1f} шт.). "
                f"Заказов в пути: {orders_in_transit:.1f} шт.",
            "ИЗБЫТОК - НЕ ЗАКАЗЫВАТЬ":
                f"Избыточные запасы ({current_stock:.1f} шт.). "
                f"Заказов в пути: {orders_in_transit:.1f} шт."
        }
        return reasons.get(status, "Рекомендация на основе анализа спроса, запасов и заказов в пути.")

class ProductAnalytics:
    def __init__(self, df, product_col='product_code', revenue_col='revenue',
                 quantity_col='quantity', profit_col='profit', date_col='date'):
        self.df = df.copy()
        for col in [revenue_col, quantity_col, profit_col]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0)
        self.product_col = product_col
        self.revenue_col = revenue_col
        self.quantity_col = quantity_col
        self.profit_col = profit_col
        self.date_col = date_col
        if date_col in self.df.columns and date_col in self.df:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')

    def abc_analysis(self, a_threshold=0.8, b_threshold=0.95):
        try:
            if self.df.empty:
                return pd.DataFrame({
                    self.product_col: [],
                    self.revenue_col: [],
                    'revenue_share': [],
                    'cumulative_share': [],
                    'abc_category': []
                })
            product_revenue = self.df.groupby(self.product_col)[self.revenue_col].sum().reset_index()
            product_revenue = product_revenue[product_revenue[self.revenue_col] > 0]
            if product_revenue.empty:
                return pd.DataFrame({
                    self.product_col: [],
                    self.revenue_col: [],
                    'revenue_share': [],
                    'cumulative_share': [],
                    'abc_category': []
                })
            product_revenue = product_revenue.sort_values(self.revenue_col, ascending=False)
            total_revenue = product_revenue[self.revenue_col].sum()
            product_revenue['revenue_share'] = product_revenue[self.revenue_col] / total_revenue
            product_revenue['cumulative_share'] = product_revenue['revenue_share'].cumsum()
            def assign_category(cum_share):
                if cum_share <= a_threshold:
                    return 'A'
                elif cum_share <= b_threshold:
                    return 'B'
                else:
                    return 'C'
            product_revenue['abc_category'] = product_revenue['cumulative_share'].apply(assign_category)
            return product_revenue
        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame({
                self.product_col: [],
                self.revenue_col: [],
                'revenue_share': [],
                'cumulative_share': [],
                'abc_category': []
            })

    def xyz_analysis(self, period='M', cv_threshold_low=0.1, cv_threshold_medium=0.25):
        try:
            if self.df.empty or self.date_col not in self.df.columns:
                return pd.DataFrame({
                    self.product_col: [],
                    'total_quantity': [],
                    'avg_quantity': [],
                    'std_quantity': [],
                    'cv': [],
                    'xyz_category': [],
                    'n_periods': []
                })
            df_copy = self.df.copy()
            df_copy = df_copy.dropna(subset=[self.date_col])
            if df_copy.empty:
                return pd.DataFrame({
                    self.product_col: [],
                    'total_quantity': [],
                    'avg_quantity': [],
                    'std_quantity': [],
                    'cv': [],
                    'xyz_category': [],
                    'n_periods': []
                })
            if period == 'M':
                df_copy['period'] = df_copy[self.date_col].dt.to_period('M')
            elif period == 'W':
                df_copy['period'] = df_copy[self.date_col].dt.to_period('W')
            else:
                raise ValueError("Период должен быть 'M' или 'W'")
            product_period = df_copy.groupby([self.product_col, 'period'])[self.quantity_col].sum().reset_index()
            period_counts = product_period.groupby(self.product_col)['period'].nunique()
            valid_products = period_counts[period_counts >= 2].index.tolist()
            if not valid_products:
                return pd.DataFrame({
                    self.product_col: [],
                    'total_quantity': [],
                    'avg_quantity': [],
                    'std_quantity': [],
                    'cv': [],
                    'xyz_category': [],
                    'n_periods': []
                })
            results = []
            for product in valid_products:
                product_data = product_period[product_period[self.product_col] == product][self.quantity_col]
                if len(product_data) >= 2:
                    mean_val = product_data.mean()
                    if mean_val > 0:
                        cv = product_data.std() / mean_val
                    else:
                        cv = 0
                else:
                    cv = 0
                if cv <= cv_threshold_low:
                    xyz_category = 'X'
                elif cv <= cv_threshold_medium:
                    xyz_category = 'Y'
                else:
                    xyz_category = 'Z'
                results.append({
                    self.product_col: product,
                    'total_quantity': product_data.sum(),
                    'avg_quantity': product_data.mean() if len(product_data) > 0 else 0,
                    'std_quantity': product_data.std() if len(product_data) > 1 else 0,
                    'cv': cv,
                    'xyz_category': xyz_category,
                    'n_periods': len(product_data)
                })
            xyz_df = pd.DataFrame(results)
            return xyz_df
        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame({
                self.product_col: [],
                'total_quantity': [],
                'avg_quantity': [],
                'std_quantity': [],
                'cv': [],
                'xyz_category': [],
                'n_periods': []
            })

    def seasonality_analysis(self, product_id=None, group_by='month'):
        try:
            if self.date_col not in self.df.columns or self.date_col not in self.df:
                return self._create_empty_seasonality_df()
            if product_id is not None:
                df_filtered = self.df[self.df[self.product_col] == product_id].copy()
            else:
                df_filtered = self.df.copy()
            if df_filtered.empty:
                return self._create_empty_seasonality_df()
            if group_by == 'month':
                df_filtered['period'] = df_filtered[self.date_col].dt.month
                periods = list(range(1, 13))
                period_names = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                                'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
            elif group_by == 'week':
                df_filtered['period'] = df_filtered[self.date_col].dt.isocalendar().week
                periods = list(range(1, 53))
                period_names = [f'Неделя {i}' for i in periods]
            elif group_by == 'quarter':
                df_filtered['period'] = df_filtered[self.date_col].dt.quarter
                periods = [1, 2, 3, 4]
                period_names = ['Q1', 'Q2', 'Q3', 'Q4']
            else:
                raise ValueError("group_by должен быть 'month', 'week' или 'quarter'")
            period_df = pd.DataFrame({
                'period': periods,
                'period_name': period_names
            })
            grouped = df_filtered.groupby('period').agg({
                self.revenue_col: 'sum',
                self.quantity_col: 'sum',
                self.profit_col: 'sum'
            }).reset_index()
            result = period_df.merge(grouped, on='period', how='left')
            result[self.revenue_col] = result[self.revenue_col].fillna(0)
            result[self.quantity_col] = result[self.quantity_col].fillna(0)
            result[self.profit_col] = result[self.profit_col].fillna(0)
            total_revenue = result[self.revenue_col].sum()
            total_quantity = result[self.quantity_col].sum()
            if total_revenue > 0:
                result['revenue_share'] = (result[self.revenue_col] / total_revenue * 100).round(2)
            else:
                result['revenue_share'] = 0
            if total_quantity > 0:
                result['quantity_share'] = (result[self.quantity_col] / total_quantity * 100).round(2)
            else:
                result['quantity_share'] = 0
            result = result.sort_values('period')
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._create_empty_seasonality_df()

    def _create_empty_seasonality_df(self):
        return pd.DataFrame({
            'period': [],
            'period_name': [],
            'revenue': [],
            'quantity': [],
            'profit': [],
            'revenue_share': [],
            'quantity_share': []
        })

    def top_products_by_profitability(self, n=10, metric='profit_margin'):
        try:
            product_stats = self.df.groupby(self.product_col).agg({
                self.revenue_col: 'sum',
                self.quantity_col: 'sum',
                self.profit_col: 'sum'
            }).reset_index()
            product_stats['profit_margin'] = 0.0
            mask_revenue = product_stats[self.revenue_col] > 0
            product_stats.loc[mask_revenue, 'profit_margin'] = (
                    product_stats.loc[mask_revenue, self.profit_col] /
                    product_stats.loc[mask_revenue, self.revenue_col] * 100
            ).round(2)
            product_stats['profit_per_unit'] = 0.0
            mask_quantity = product_stats[self.quantity_col] > 0
            product_stats.loc[mask_quantity, 'profit_per_unit'] = (
                    product_stats.loc[mask_quantity, self.profit_col] /
                    product_stats.loc[mask_quantity, self.quantity_col]
            ).round(2)
            product_stats['avg_price'] = 0.0
            product_stats.loc[mask_quantity, 'avg_price'] = (
                    product_stats.loc[mask_quantity, self.revenue_col] /
                    product_stats.loc[mask_quantity, self.quantity_col]
            ).round(2)
            if metric == 'profit_margin':
                top_products = product_stats.sort_values('profit_margin', ascending=False).head(n)
            elif metric == 'total_profit':
                top_products = product_stats.sort_values(self.profit_col, ascending=False).head(n)
            elif metric == 'profit_per_unit':
                top_products = product_stats.sort_values('profit_per_unit', ascending=False).head(n)
            else:
                raise ValueError("metric должен быть 'profit_margin', 'total_profit' или 'profit_per_unit'")
            return top_products
        except Exception as e:
            return pd.DataFrame({
                self.product_col: [],
                self.revenue_col: [],
                self.quantity_col: [],
                self.profit_col: [],
                'profit_margin': [],
                'profit_per_unit': [],
                'avg_price': []
            })