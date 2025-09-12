import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.dummy import DummyRegressor
import ephem
import swisseph as swe
import warnings
warnings.filterwarnings('ignore')

# Crime severity mapping by CVA rank
SEVERITY_MAP = {
    'HOMICIDE': 10,
    'CRIMINAL SEXUAL ASSAULT': 9,
    'SEX OFFENSE': 7,
    'ROBBERY': 5,
    'BATTERY': 6,
    'ASSAULT': 6,
    'BURGLARY': 4,
    'MOTOR VEHICLE THEFT': 4,
    'THEFT': 4,
    'ARSON': 4,
    'CRIMINAL DAMAGE': 3,
    'NARCOTICS': 3,
    'STALKING': 4,
    'OFFENSE INVOLVING CHILDREN': 8,
    'HUMAN TRAFFICKING': 9,
    'KIDNAPPING': 8,
    'DECEPTIVE PRACTICE': 2,
    'OTHER OFFENSE': 1,
    'WEAPONS VIOLATION': 3,
    'CRIMINAL TRESPASS': 3,
    'INTERFERENCE WITH PUBLIC OFFICER': 1,
    'LIQUOR LAW VIOLATION': 1,
    'PUBLIC PEACE VIOLATION': 2,
    'CRIMINAL SEXUAL ASSAULT': 9,
    'PROSTITUTION': 2,
    'CRIMINAL DAMAGE': 3,
    'CRIMINAL TRESPASS': 3,
    'PUBLIC PEACE VIOLATION': 2,
    'CRIMINAL SEXUAL ASSAULT': 9,
    'PROSTITUTION': 2,
    'GAMBLING': 1,
    'INTIMIDATION': 4,
    'CRIMINAL ABORTION': 5,
    'CONCEALED CARRY LICENSE VIOLATION': 2,
    'RITUALISM': 1,
    'NON-CRIMINAL': 1,
    'OTHER NARCOTIC VIOLATION': 3,
    'CRIMINAL DAMAGE': 3,
    'OBSCENITY': 2,
    'HUMAN TRAFFICKING': 9,
    'DOMESTIC VIOLENCE': 5,
    'OTHER ASSAULT': 4,
    'KIDNAPPING': 8,
    'ARSON': 4,
    'MOTOR VEHICLE THEFT': 4,
    'THEFT OF SERVICE': 2,
    'STALKING': 4,
    'CRIMINAL TRESPASS TO VEHICLE': 2,
    'INTERFERENCE WITH PUBLIC OFFICER': 1,
    'LIQUOR LAW VIOLATION': 1,
    'PUBLIC PEACE VIOLATION': 2,
    'OFFENSE INVOLVING CHILDREN': 8,
    'HUMAN TRAFFICKING': 9,
    'KIDNAPPING': 8,
    'DECEPTIVE PRACTICE': 2,
    'OTHER OFFENSE': 1,
    'WEAPONS VIOLATION': 3,
    'CRIMINAL TRESPASS': 3,
    'INTERFERENCE WITH PUBLIC OFFICER': 1,
    'LIQUOR LAW VIOLATION': 1,
    'PUBLIC PEACE VIOLATION': 2,
    'CRIMINAL SEXUAL ASSAULT': 9,
    'PROSTITUTION': 2,
    'CRIMINAL DAMAGE': 3,
    'CRIMINAL TRESPASS': 3,
    'PUBLIC PEACE VIOLATION': 2,
    'CRIMINAL SEXUAL ASSAULT': 9,
    'PROSTITUTION': 2,
    'GAMBLING': 1,
    'INTIMIDATION': 4,
    'CRIMINAL ABORTION': 5,
    'CONCEALED CARRY LICENSE VIOLATION': 2,
    'RITUALISM': 1,
    'NON-CRIMINAL': 1,
    'OTHER NARCOTIC VIOLATION': 3,
    'CRIMINAL DAMAGE': 3,
    'OBSCENITY': 2,
    'HUMAN TRAFFICKING': 9,
    'DOMESTIC VIOLENCE': 5,
    'OTHER ASSAULT': 4,
    'KIDNAPPING': 8,
    'ARSON': 4,
    'MOTOR VEHICLE THEFT': 4,
    'THEFT OF SERVICE': 2,
    'STALKING': 4,
    'CRIMINAL TRESPASS TO VEHICLE': 2,
    'PUBLIC PEACE VIOLATION': 2,
    'HUMAN TRAFFICKING': 9,
    'CRIMINAL SEXUAL ASSAULT': 9
}

class CrimeDataFetcher:
    """Fetch and process Chicago crime data"""
    
    def __init__(self):
        self.chicago_info = {
            'url': 'https://data.cityofchicago.org/resource/ijzp-q8t2.json',
            'lat': 41.8781,
            'lon': -87.6298,
            'timezone': 'America/Chicago'
        }

    def fetch_chicago_data(self, start_year=2020, end_year=2024):
        """Fetch Chicago crime data for specified years"""
        print("="*70)
        print(f"FETCHING CHICAGO CRIME DATA ({start_year}-{end_year})")
        print("="*70)

        all_data = []
        batch_size = 100000
        offset = 0
        batch_num = 1

        while True:
            params = {
                '$limit': batch_size,
                '$offset': offset,
                '$where': f"date >= '{start_year}-01-01T00:00:00' AND date <= '{end_year}-12-31T23:59:59'",
                '$select': 'date,primary_type'
            }

            try:
                print(f"Fetching batch {batch_num} (offset: {offset:,})...")
                response = requests.get(self.chicago_info['url'], params=params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        print("✅ No more data to fetch")
                        break
                        
                    print(f"  ✅ Retrieved {len(data):,} records")
                    all_data.extend(data)
                    
                    if len(data) < batch_size:
                        print("✅ Reached end of dataset")
                        break
                        
                    offset += batch_size
                    batch_num += 1
                else:
                    print(f"❌ HTTP Error: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                break

        print(f"\n{'='*70}")
        print(f"TOTAL RECORDS FETCHED: {len(all_data):,}")
        print(f"{'='*70}\n")
        
        return all_data

    def process_data(self, data):
        """Process crime data into hourly severity totals"""
        print("Processing crime data...")
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        # Map severity
        df['crime_type'] = df['primary_type'].str.upper().str.strip()
        df['severity'] = df['crime_type'].map(SEVERITY_MAP).fillna(3)
        
        # Create hourly bins
        df['hour'] = df['datetime'].dt.floor('H')
        
        # Aggregate different severity metrics by hour
        hourly_data = df.groupby('hour').agg(
            mean_severity=('severity', 'mean'),
            median_severity=('severity', 'median'),
            sum_severity=('severity', 'sum'),
            crime_count=('crime_type', 'count')
        ).reset_index()

        hourly_data.columns = ['datetime', 'mean_severity', 'median_severity', 'sum_severity', 'crime_count']
        
        print(f"✅ Processed {len(df):,} crimes into {len(hourly_data):,} hourly periods")
        print(f"   Date range: {hourly_data['datetime'].min()} to {hourly_data['datetime'].max()}")
        
        return hourly_data

class AstronomicalCalculator:
    """Advanced astronomical calculations using Swiss Ephemeris"""
    
    def __init__(self, lat=41.8781, lon=-87.6298, timezone='America/Chicago'):
        self.lat = lat
        self.lon = lon
        self.timezone = timezone
        
        # Initialize Swiss Ephemeris
        swe.set_ephe_path(None)  # Use built-in ephemeris
        
        # Planet constants
        self.PLANETS = {
            'sun': swe.SUN,
            'moon': swe.MOON,
            'mercury': swe.MERCURY,
            'venus': swe.VENUS,
            'mars': swe.MARS,
            'jupiter': swe.JUPITER,
            'saturn': swe.SATURN,
            'uranus': swe.URANUS,
            'neptune': swe.NEPTUNE,
            'pluto': swe.PLUTO,
            'north_node': swe.MEAN_NODE,
            'south_node': swe.MEAN_NODE  # Will calculate as opposite
        }
        
        # Major aspects (degrees)
        self.ASPECTS = {
            'conjunction': 0,
            'sextile': 60,
            'square': 90,
            'trine': 120,
            'opposition': 180
        }
        
        # Orbs for aspects (degrees)
        self.ORBS = {
            'conjunction': 8,
            'sextile': 6,
            'square': 8,
            'trine': 8,
            'opposition': 8
        }

    def julian_day(self, dt):
        """Convert datetime to Julian day"""
        return swe.julday(dt.year, dt.month, dt.day, 
                         dt.hour + dt.minute/60.0 + dt.second/3600.0)

    def get_planet_position(self, jd, planet):
        """Get planet position in ecliptic longitude"""
        if planet == 'south_node':
            # South node is 180° opposite to north node
            north_pos = swe.calc_ut(jd, swe.MEAN_NODE)[0][0]
            return (north_pos + 180) % 360
        else:
            return swe.calc_ut(jd, self.PLANETS[planet])[0][0]

    def get_ascendant(self, jd):
        """Calculate accurate ascendant using Swiss Ephemeris"""
        houses = swe.houses(jd, self.lat, self.lon, b'P')  # Placidus house system
        return houses[1][0]  # Ascendant is the first house cusp

    def calculate_aspect(self, pos1, pos2, aspect_degrees, orb):
        """Calculate if two positions form an aspect"""
        diff = abs(pos1 - pos2)
        if diff > 180:
            diff = 360 - diff
            
        aspect_diff = abs(diff - aspect_degrees)
        if aspect_diff <= orb:
            return True, orb - aspect_diff  # Return strength (tighter = stronger)
        return False, 0

    def get_all_aspects(self, positions):
        """Calculate all major aspects between planets"""
        aspects = {}
        planet_names = list(positions.keys())
        
        for i, planet1 in enumerate(planet_names):
            for planet2 in planet_names[i+1:]:
                for aspect_name, aspect_degrees in self.ASPECTS.items():
                    orb = self.ORBS[aspect_name]
                    is_aspect, strength = self.calculate_aspect(
                        positions[planet1], positions[planet2], 
                        aspect_degrees, orb
                    )
                    
                    if is_aspect:
                        aspect_key = f"{planet1}_{aspect_name}_{planet2}"
                        aspects[aspect_key] = strength
        
        return aspects

    def is_eclipse_period(self, jd):
        """Check if date is within eclipse influence (±13 days from eclipse)"""
        # This is a simplified eclipse checker
        # For production, you'd want a more comprehensive eclipse database
        
        # Some major eclipses 2020-2024 (Julian days)
        eclipse_dates = [
            2458849.5,  # Jan 10, 2020 Lunar Eclipse
            2458940.5,  # Jun 5, 2020 Lunar Eclipse
            2458953.5,  # Jun 21, 2020 Solar Eclipse
            2459034.5,  # Dec 14, 2020 Solar Eclipse
            2459363.5,  # May 26, 2021 Lunar Eclipse
            2459377.5,  # Jun 10, 2021 Solar Eclipse
            2459522.5,  # Nov 19, 2021 Lunar Eclipse
            2459597.5,  # Apr 30, 2022 Solar Eclipse
            2459611.5,  # May 16, 2022 Lunar Eclipse
            2459858.5,  # Oct 25, 2022 Solar Eclipse
            2459871.5,  # Nov 8, 2022 Lunar Eclipse
            2460064.5,  # Apr 20, 2023 Solar Eclipse
            2460079.5,  # May 5, 2023 Lunar Eclipse
            2460203.5,  # Oct 14, 2023 Solar Eclipse
            2460232.5,  # Oct 28, 2023 Lunar Eclipse
            2460409.5,  # Apr 8, 2024 Solar Eclipse
            2460423.5,  # Sep 18, 2024 Lunar Eclipse
            2460565.5   # Oct 2, 2024 Solar Eclipse
        ]
        
        eclipse_influence = 13.0  # 13 days before/after eclipse
        
        for eclipse_jd in eclipse_dates:
            if abs(jd - eclipse_jd) <= eclipse_influence:
                return True, eclipse_influence - abs(jd - eclipse_jd)
        
        return False, 0

    def mercury_retrograde_periods(self):
        """Mercury retrograde periods 2020-2024"""
        return [
            (datetime(2020, 2, 17), datetime(2020, 3, 10)),
            (datetime(2020, 6, 18), datetime(2020, 7, 12)),
            (datetime(2020, 10, 14), datetime(2020, 11, 3)),
            (datetime(2021, 1, 30), datetime(2021, 2, 21)),
            (datetime(2021, 5, 30), datetime(2021, 6, 23)),
            (datetime(2021, 9, 27), datetime(2021, 10, 18)),
            (datetime(2022, 1, 14), datetime(2022, 2, 4)),
            (datetime(2022, 5, 10), datetime(2022, 6, 3)),
            (datetime(2022, 9, 10), datetime(2022, 10, 2)),
            (datetime(2022, 12, 29), datetime(2023, 1, 18)),
            (datetime(2023, 4, 21), datetime(2023, 5, 15)),
            (datetime(2023, 8, 23), datetime(2023, 9, 15)),
            (datetime(2023, 12, 13), datetime(2024, 1, 2)),
            (datetime(2024, 4, 2), datetime(2024, 4, 25)),
            (datetime(2024, 8, 5), datetime(2024, 8, 28)),
            (datetime(2024, 11, 26), datetime(2024, 12, 15))
        ]

    def is_mercury_retrograde(self, dt):
        """Check if Mercury is retrograde at given datetime"""
        for start, end in self.mercury_retrograde_periods():
            if start <= dt <= end:
                return True
        return False

    def moon_phase(self, jd):
        """Calculate moon phase (0 = new moon, 1 = full moon)"""
        sun_pos = self.get_planet_position(jd, 'sun')
        moon_pos = self.get_planet_position(jd, 'moon')
        
        phase_angle = (moon_pos - sun_pos) % 360
        if phase_angle > 180:
            phase_angle = 360 - phase_angle
            
        return phase_angle / 180.0

    def calculate_astronomical_features(self, dt):
        """Calculate all astronomical features for given datetime"""
        jd = self.julian_day(dt)
        
        features = {}
        
        # 1. Planet positions
        positions = {}
        for planet in self.PLANETS.keys():
            if planet != 'south_node':  # Handle south node separately
                positions[planet] = self.get_planet_position(jd, planet)
        
        # Add south node
        positions['south_node'] = self.get_planet_position(jd, 'south_node')
        
        # Store positions as features
        for planet, position in positions.items():
            features[f'{planet}_longitude'] = position
        
        # 2. Accurate Ascendant
        features['ascendant'] = self.get_ascendant(jd)
        
        # 3. Moon phase
        features['moon_phase'] = self.moon_phase(jd)
        
        # 4. Mercury retrograde
        features['mercury_retrograde'] = int(self.is_mercury_retrograde(dt))
        
        # 5. Eclipse influence
        is_eclipse, eclipse_strength = self.is_eclipse_period(jd)
        features['eclipse_influence'] = int(is_eclipse)
        features['eclipse_strength'] = eclipse_strength
        
        # 6. Major aspects between all planets
        aspects = self.get_all_aspects(positions)
        '''
        # Create binary features for each possible aspect
        for planet1 in positions.keys():
            for planet2 in positions.keys():
                if planet1 < planet2:  # Avoid duplicates
                    for aspect_name in self.ASPECTS.keys():
                        aspect_key = f"{planet1}_{aspect_name}_{planet2}"
                        features[aspect_key] = aspects.get(aspect_key, 0)
        
        # 7. Special lunar aspects (Moon to other planets)
        for planet in ['sun', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']:
            for aspect_name in ['conjunction', 'square', 'opposition']:
                aspect_key = f"moon_{aspect_name}_{planet}"
                if aspect_key in aspects:
                    features[f"lunar_{aspect_name}_{planet}"] = aspects[aspect_key]
                else:
                    features[f"lunar_{aspect_name}_{planet}"] = 0
        
        # 8. Nodal aspects (especially to personal planets)
        for planet in ['sun', 'moon', 'mercury', 'venus', 'mars']:
            for node in ['north_node', 'south_node']:
                for aspect_name in ['conjunction', 'square', 'opposition']:
                    aspect_key = f"{planet}_{aspect_name}_{node}"
                    if aspect_key in aspects:
                        features[f"nodal_{planet}_{aspect_name}"] = aspects[aspect_key]
                    else:
                        features[f"nodal_{planet}_{aspect_name}"] = 0
        
        # 9. Hard aspects count (squares and oppositions)
        hard_aspects = 0
        for aspect_key, strength in aspects.items():
            if 'square' in aspect_key or 'opposition' in aspect_key:
                hard_aspects += strength
        features['total_hard_aspects'] = hard_aspects
        
        # 10. Harmonious aspects count (trines and sextiles)
        harmonious_aspects = 0
        for aspect_key, strength in aspects.items():
            if 'trine' in aspect_key or 'sextile' in aspect_key:
                harmonious_aspects += strength
        features['total_harmonious_aspects'] = harmonious_aspects
        '''
        return features

class ChicagoAstroCrimePredictor:
    """Chicago crime predictor using advanced astronomical features"""
    
    def __init__(self):
        self.fetcher = CrimeDataFetcher()
        self.astro_calc = AstronomicalCalculator()
        self.models = {}
        self.results = {}
        self.feature_names = []

    def prepare_data(self, start_year=2020, end_year=2024):
        """Prepare complete dataset with astronomical features"""
        print("\n" + "="*70)
        print("ASTRONOMICAL CRIME PREDICTION DATA PREPARATION")
        print("="*70)
        
        # Fetch and process crime data
        crime_data = self.fetcher.fetch_chicago_data(start_year, end_year)
        if not crime_data:
            return None
            
        hourly_df = self.fetcher.process_data(crime_data)
        if hourly_df.empty:
            return None
        
        # Calculate astronomical features
        print(f"\nCalculating astronomical features for {len(hourly_df):,} hours...")
        print("This will take several minutes for accurate calculations...\n")
        
        astro_features = []
        total_hours = len(hourly_df)
        
        for idx, row in hourly_df.iterrows():
            if idx % 1000 == 0:
                progress = (idx / total_hours) * 100
                print(f"Progress: {idx:,}/{total_hours:,} ({progress:.1f}%)")
            
            features = self.astro_calc.calculate_astronomical_features(row['datetime'])
            features['datetime'] = row['datetime']
            features['mean_severity'] = row['mean_severity']
            features['median_severity'] = row['median_severity']
            features['sum_severity'] = row['sum_severity']
            features['crime_count'] = row['crime_count']
            
            astro_features.append(features)
        
        df = pd.DataFrame(astro_features)
        
        print(f"\n✅ Complete dataset prepared:")
        print(f"   Shape: {df.shape}")
        print(f"   Astronomical features: {df.shape[1] - 5}")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df

    def train_models(self, df, target_variable):
        """Train models using astronomical features for a specific target"""
        print("\n" + "="*70)
        print(f"ASTRONOMICAL MODEL TRAINING FOR TARGET: {target_variable}")
        print("="*70)
        
        # Prepare features
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'mean_severity', 'median_severity', 'sum_severity', 'crime_count']]
        
        X = df[feature_cols]
        y = df[target_variable]
        
        self.feature_names = feature_cols
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X):,}")
        
        # Time series split (important for temporal data)
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(X))[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Feature selection to reduce overfitting
        print(f"Original features: {X_train.shape[1]}")
        n_features = min(50, X_train.shape[1])
        selector = SelectKBest(f_regression, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected features: {len(selected_features)}")
        
        X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

        print(f"Training set: {len(X_train):,}")
        print(f"Test set: {len(X_test):,}")
        
        models = {
            'Mean Baseline': DummyRegressor(strategy='mean'),
            'Median Baseline': DummyRegressor(strategy='median'),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        print("\nTraining models:")
        for name, model in models.items():
            print(f"\n{name}...")
            
            if 'Regression' in name or 'Baseline' in name:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if name == 'Quantile Baseline':
                    model = DummyRegressor(strategy='quantile', quantile=0.8)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                y_train_pred = model.predict(X_train_scaled)
                train_r2 = r2_score(y_train, y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                
                cv_scores = []
                for train_idx_cv, val_idx_cv in tscv.split(X_train):
                    X_cv_train = X_train.iloc[train_idx_cv]
                    X_cv_val = X_train.iloc[val_idx_cv]
                    y_cv_train = y_train.iloc[train_idx_cv]
                    y_cv_val = y_train.iloc[val_idx_cv]
                    
                    scaler_cv = StandardScaler()
                    X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
                    X_cv_val_scaled = scaler_cv.transform(X_cv_val)
                    
                    if name == 'Ridge Regression':
                        model_cv = Ridge(alpha=1.0)
                    elif name == 'Linear Regression':
                        model_cv = LinearRegression()
                    elif name == 'Mean Baseline':
                        model_cv = DummyRegressor(strategy='mean')
                    elif name == 'Median Baseline':
                        model_cv = DummyRegressor(strategy='median')
                    
                    model_cv.fit(X_cv_train_scaled, y_cv_train)
                    y_cv_pred = model_cv.predict(X_cv_val_scaled)
                    cv_scores.append(-mean_squared_error(y_cv_val, y_cv_pred))
                
                cv_scores = np.array(cv_scores)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                y_train_pred = model.predict(X_train)
                train_r2 = r2_score(y_train, y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=tscv, scoring='neg_mean_squared_error'
                )
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'cv_rmse': cv_rmse,
                'cv_std': cv_std,
                'target': target_variable
            }
            
            print(f"  Test R² Score: {r2:.4f}")
            print(f"  Train R² Score: {train_r2:.4f}")
            print(f"  Test RMSE: {rmse:.4f}")
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  CV RMSE: {cv_rmse:.4f} (±{cv_std:.4f})")
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                results[name]['feature_importance'] = importance_df
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        return results

    def print_results(self):
        """Print comprehensive results with astronomical insights"""
        # This function will be called separately for each target to print detailed reports
        pass
    
    def print_final_summary(self, all_results):
        """Print final summary of the best model and target combination"""
        print("\n" + "="*80)
        print("FINAL ANALYSIS: BEST MODEL ACROSS ALL TARGETS")
        print("="*80)
        
        best_overall = None
        best_r2 = -float('inf')
        
        print("\n{:<15} | {:<20} | {:<10} | {:<10}".format(
            'Target', 'Model', 'Test R²', 'Test RMSE'
        ))
        print("-" * 60)
        
        for target, results in all_results.items():
            sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
            best_model_name, best_metrics = sorted_models[0]
            
            print("{:<15} | {:<20} | {:<10.4f} | {:<10.4f}".format(
                target, best_model_name, best_metrics['r2'], best_metrics['rmse']
            ))
            
            if best_metrics['r2'] > best_r2:
                best_r2 = best_metrics['r2']
                best_overall = (target, best_model_name, best_metrics)
                
        if best_overall:
            target, name, metrics = best_overall
            print("\n" + "="*80)
            print("OVERALL BEST MODEL: {} for predicting {}".format(name, target))
            print("="*80)
            print("Test R² Score: {:.6f}".format(metrics['r2']))
            print("Test RMSE: {:.6f}".format(metrics['rmse']))
            print("Cross-Val RMSE: {:.6f}".format(metrics['cv_rmse']))
            
            if 'feature_importance' in metrics:
                importance_df = metrics['feature_importance']
                print("\n🌟 TOP 10 MOST IMPORTANT ASTRONOMICAL FEATURES:")
                print("-" * 60)
                for idx, row in importance_df.head(10).iterrows():
                    importance_pct = row['importance'] * 100
                    print("  • {:<35}: {:6.2f}%".format(row['feature'], importance_pct))
        else:
            print("\n❌ No results to display.")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CHICAGO CRIME SEVERITY PREDICTOR")
    print("Advanced Astronomical Feature Analysis")
    print("Swiss Ephemeris - High Precision Calculations")
    print("="*80)
    
    predictor = ChicagoAstroCrimePredictor()
    
    df = predictor.prepare_data(start_year=2022, end_year=2024)
    
    if df is not None and not df.empty:
        target_variables = ['mean_severity', 'median_severity', 'sum_severity']
        all_results = {}
        
        for target in target_variables:
            all_results[target] = predictor.train_models(df, target)
        
        predictor.print_final_summary(all_results)
        
    else:
        print("\n❌ Unable to complete analysis due to data issues")
        return None

if __name__ == "__main__":
    main()