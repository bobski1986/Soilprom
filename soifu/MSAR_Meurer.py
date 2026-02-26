"""
Integrated MSAR-Meurer Framework
=================================
Coupling earthworm population responses to chemical exposure (MSAR)
with soil structure dynamics (Meurer et al. 2020)

This framework demonstrates how chemical pollution affects:
1. Earthworm population abundance (MSAR model)
2. Earthworm biomass and bioturbation activity
3. Soil structure recovery and maintenance
4. Ecosystem service delivery

Author: Artur
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import lognorm, norm
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PART 1: SYNTHETIC EARTHWORM TOXICITY DATABASE
# ============================================================================

class SyntheticToxicityDatabase:
    """
    Generate realistic earthworm toxicity data for testing MSAR framework
    
    Based on literature distributions:
    - Spurgeon et al. (2003) - Metal toxicity patterns
    - Vijver et al. (2003) - Species sensitivity
    - PPDB database statistics
    
    Generates data for:
    - Multiple chemical classes (metals, pesticides)
    - Multiple earthworm species
    - Various endpoints (LC50, EC50 reproduction)
    - Realistic variability and uncertainty
    """
    
    def __init__(self, n_chemicals=50, seed=42):
        np.random.seed(seed)
        self.n_chemicals = n_chemicals
        self.chemicals = None
        self.toxicity_data = None
        
    def generate_chemical_library(self):
        """
        Generate synthetic chemical library with realistic properties
        
        Chemical classes:
        1. Heavy metals (Cu, Zn, Pb, Cd, Ni)
        2. Organophosphate pesticides
        3. Neonicotinoid pesticides
        4. PAHs (Polycyclic Aromatic Hydrocarbons)
        5. Herbicides
        """
        
        chemical_classes = {
            'Heavy Metals': ['Copper', 'Zinc', 'Lead', 'Cadmium', 'Nickel', 
                            'Chromium', 'Mercury'],
            'Organophosphates': ['Chlorpyrifos', 'Diazinon', 'Malathion', 
                                'Parathion', 'Dimethoate'],
            'Neonicotinoids': ['Imidacloprid', 'Thiamethoxam', 'Clothianidin',
                              'Acetamiprid', 'Thiacloprid'],
            'Herbicides': ['Atrazine', 'Glyphosate', 'Metolachlor', 
                          '2,4-D', 'Paraquat'],
            'Fungicides': ['Carbendazim', 'Benomyl', 'Mancozeb', 'Captan'],
            'Other': ['Triclosan', 'Nonylphenol', 'Bisphenol-A']
        }
        
        chemicals = []
        chem_id = 1
        
        for class_name, chem_list in chemical_classes.items():
            for chem_name in chem_list:
                if len(chemicals) >= self.n_chemicals:
                    break
                    
                chemicals.append({
                    'Chemical_ID': f'CHEM_{chem_id:03d}',
                    'Chemical_Name': chem_name,
                    'Chemical_Class': class_name,
                    'CAS_Number': f'{100000 + chem_id}-00-0',
                    'Mode_of_Action': self._assign_moa(class_name),
                    'Log_Kow': self._generate_log_kow(class_name),
                    'Molecular_Weight': np.random.uniform(100, 500)
                })
                chem_id += 1
        
        self.chemicals = pd.DataFrame(chemicals[:self.n_chemicals])
        print(f"✓ Generated {len(self.chemicals)} chemicals across {len(chemical_classes)} classes")
        return self.chemicals
    
    def _assign_moa(self, chemical_class):
        """Assign mode of action"""
        moa_map = {
            'Heavy Metals': 'Oxidative stress, enzyme inhibition',
            'Organophosphates': 'Acetylcholinesterase inhibition',
            'Neonicotinoids': 'Nicotinic acetylcholine receptor agonist',
            'PAHs': 'Narcosis, DNA damage',
            'Herbicides': 'Various (photosystem II, protein synthesis)',
            'Fungicides': 'Microtubule assembly inhibition',
            'Other': 'Various mechanisms'
        }
        return moa_map.get(chemical_class, 'Unknown')
    
    def _generate_log_kow(self, chemical_class):
        """Generate realistic log Kow values"""
        kow_ranges = {
            'Heavy Metals': (0.5, 2.0),
            'Organophosphates': (2.5, 4.5),
            'Neonicotinoids': (0.5, 1.5),
            'PAHs': (3.5, 6.5),
            'Herbicides': (1.5, 4.0),
            'Fungicides': (2.0, 4.0),
            'Other': (2.0, 5.0)
        }
        low, high = kow_ranges.get(chemical_class, (1.0, 4.0))
        return np.random.uniform(low, high)
    
    def generate_toxicity_data(self):
        """
        Generate realistic toxicity data for all chemicals
        
        Follows empirical distributions from literature:
        - LC50 values: log-normal distribution
        - EC50 typically 2-10x lower than LC50 (sublethal effects)
        - Species sensitivity varies (log-normal around mean)
        - Test duration affects observed toxicity
        
        Reference distributions:
        - Metals LC50 (14-day): median ~100 mg/kg, range 10-1000
        - Pesticides LC50 (14-day): median ~50 mg/kg, range 1-500
        - EC50 (reproduction): typically 0.2-0.5 × LC50
        """
        
        if self.chemicals is None:
            self.generate_chemical_library()
        
        species_list = [
            {'species': 'Eisenia fetida', 'sensitivity': 1.0, 'group': 'Epigeic'},
            {'species': 'Eisenia andrei', 'sensitivity': 1.1, 'group': 'Epigeic'},
            {'species': 'Lumbricus terrestris', 'sensitivity': 1.5, 'group': 'Anecic'},
            {'species': 'Lumbricus rubellus', 'sensitivity': 1.3, 'group': 'Epigeic'},
            {'species': 'Aporrectodea caliginosa', 'sensitivity': 1.4, 'group': 'Endogeic'}
        ]
        
        toxicity_records = []
        
        for _, chem in self.chemicals.iterrows():
            # Base toxicity depends on chemical class
            base_lc50 = self._get_base_toxicity(chem['Chemical_Class'])
            
            for species in species_list:
                # Species-specific LC50 (14-day acute test)
                lc50_14d = base_lc50 * species['sensitivity'] * np.random.lognormal(0, 0.3)
                
                # 28-day LC50 (typically similar to 14-day for most compounds)
                lc50_28d = lc50_14d * np.random.uniform(0.8, 1.2)
                
                # EC50 for reproduction (typically 0.2-0.5 of LC50)
                ec50_repro = lc50_14d * np.random.uniform(0.2, 0.5)
                
                # NOEC (typically 0.1-0.3 of EC50)
                noec_repro = ec50_repro * np.random.uniform(0.1, 0.3)
                
                # Add LC50 14-day record
                toxicity_records.append({
                    'Chemical_ID': chem['Chemical_ID'],
                    'Chemical_Name': chem['Chemical_Name'],
                    'Chemical_Class': chem['Chemical_Class'],
                    'CAS_Number': chem['CAS_Number'],
                    'Species': species['species'],
                    'Ecological_Group': species['group'],
                    'Endpoint': 'LC50',
                    'Effect': 'Mortality',
                    'Test_Duration_Days': 14,
                    'Concentration_mg_kg': lc50_14d,
                    'Lower_CI': lc50_14d * 0.7,
                    'Upper_CI': lc50_14d * 1.3,
                    'Soil_Type': 'OECD Artificial',
                    'pH': 6.0 + np.random.uniform(-0.3, 0.3),
                    'Organic_Matter_percent': 10.0 + np.random.uniform(-2, 2),
                    'Test_Guideline': np.random.choice(['OECD 207', 'ISO 11268-1']),
                    'Quality_Score': np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                })
                
                # Add LC50 28-day record (for some chemicals)
                if np.random.random() > 0.3:  # 70% have 28-day data
                    toxicity_records.append({
                        'Chemical_ID': chem['Chemical_ID'],
                        'Chemical_Name': chem['Chemical_Name'],
                        'Chemical_Class': chem['Chemical_Class'],
                        'CAS_Number': chem['CAS_Number'],
                        'Species': species['species'],
                        'Ecological_Group': species['group'],
                        'Endpoint': 'LC50',
                        'Effect': 'Mortality',
                        'Test_Duration_Days': 28,
                        'Concentration_mg_kg': lc50_28d,
                        'Lower_CI': lc50_28d * 0.7,
                        'Upper_CI': lc50_28d * 1.3,
                        'Soil_Type': 'OECD Artificial',
                        'pH': 6.0 + np.random.uniform(-0.3, 0.3),
                        'Organic_Matter_percent': 10.0 + np.random.uniform(-2, 2),
                        'Test_Guideline': np.random.choice(['OECD 207', 'ISO 11268-1']),
                        'Quality_Score': np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                    })
                
                # Add EC50 reproduction record
                toxicity_records.append({
                    'Chemical_ID': chem['Chemical_ID'],
                    'Chemical_Name': chem['Chemical_Name'],
                    'Chemical_Class': chem['Chemical_Class'],
                    'CAS_Number': chem['CAS_Number'],
                    'Species': species['species'],
                    'Ecological_Group': species['group'],
                    'Endpoint': 'EC50',
                    'Effect': 'Reproduction (cocoons)',
                    'Test_Duration_Days': np.random.choice([28, 56]),
                    'Concentration_mg_kg': ec50_repro,
                    'Lower_CI': ec50_repro * 0.7,
                    'Upper_CI': ec50_repro * 1.3,
                    'Soil_Type': 'OECD Artificial',
                    'pH': 6.0 + np.random.uniform(-0.3, 0.3),
                    'Organic_Matter_percent': 10.0 + np.random.uniform(-2, 2),
                    'Test_Guideline': np.random.choice(['OECD 222', 'ISO 11268-2']),
                    'Quality_Score': np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                })
                
                # Add NOEC reproduction record (for some)
                if np.random.random() > 0.4:  # 60% have NOEC
                    toxicity_records.append({
                        'Chemical_ID': chem['Chemical_ID'],
                        'Chemical_Name': chem['Chemical_Name'],
                        'Chemical_Class': chem['Chemical_Class'],
                        'CAS_Number': chem['CAS_Number'],
                        'Species': species['species'],
                        'Ecological_Group': species['group'],
                        'Endpoint': 'NOEC',
                        'Effect': 'Reproduction (cocoons)',
                        'Test_Duration_Days': np.random.choice([28, 56]),
                        'Concentration_mg_kg': noec_repro,
                        'Lower_CI': np.nan,
                        'Upper_CI': np.nan,
                        'Soil_Type': 'OECD Artificial',
                        'pH': 6.0 + np.random.uniform(-0.3, 0.3),
                        'Organic_Matter_percent': 10.0 + np.random.uniform(-2, 2),
                        'Test_Guideline': np.random.choice(['OECD 222', 'ISO 11268-2']),
                        'Quality_Score': np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                    })
        
        self.toxicity_data = pd.DataFrame(toxicity_records)
        
        print(f"\n✓ Generated toxicity database:")
        print(f"  Total records: {len(self.toxicity_data)}")
        print(f"  Chemicals: {self.toxicity_data['Chemical_Name'].nunique()}")
        print(f"  Species: {self.toxicity_data['Species'].nunique()}")
        print(f"  LC50 records: {len(self.toxicity_data[self.toxicity_data['Endpoint']=='LC50'])}")
        print(f"  EC50 records: {len(self.toxicity_data[self.toxicity_data['Endpoint']=='EC50'])}")
        
        return self.toxicity_data
    
    def _get_base_toxicity(self, chemical_class):
        """
        Get base LC50 value for chemical class (mg/kg dry soil)
        
        Based on literature meta-analyses:
        - Spurgeon & Hopkin (1996) - Metal toxicity to earthworms
        - Pelosi et al. (2013) - Pesticide effects on earthworms
        """
        base_toxicity = {
            'Heavy Metals': np.random.lognormal(np.log(100), 0.8),  # Median ~100, range 10-1000
            'Organophosphates': np.random.lognormal(np.log(50), 0.7),  # Median ~50, range 5-500
            'Neonicotinoids': np.random.lognormal(np.log(30), 0.6),   # Median ~30, range 5-200
            'PAHs': np.random.lognormal(np.log(80), 0.7),             # Median ~80, range 10-600
            'Herbicides': np.random.lognormal(np.log(200), 0.8),      # Median ~200, range 20-2000
            'Fungicides': np.random.lognormal(np.log(40), 0.7),       # Median ~40, range 5-400
            'Other': np.random.lognormal(np.log(100), 0.8)
        }
        return base_toxicity.get(chemical_class, 100.0)
    
    def export_database(self, filename='synthetic_earthworm_toxicity.xlsx'):
        """Export synthetic database to Excel"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.chemicals.to_excel(writer, sheet_name='Chemicals', index=False)
            self.toxicity_data.to_excel(writer, sheet_name='Toxicity_Data', index=False)
            
            # Summary sheet
            summary = pd.DataFrame({
                'Metric': [
                    'Total chemicals',
                    'Total toxicity records',
                    'Species tested',
                    'Chemical classes',
                    'LC50 (14-day) records',
                    'LC50 (28-day) records',
                    'EC50 (reproduction) records',
                    'NOEC records',
                    'Median LC50 (mg/kg)',
                    'Median EC50 (mg/kg)'
                ],
                'Value': [
                    len(self.chemicals),
                    len(self.toxicity_data),
                    self.toxicity_data['Species'].nunique(),
                    self.chemicals['Chemical_Class'].nunique(),
                    len(self.toxicity_data[(self.toxicity_data['Endpoint']=='LC50') & 
                                           (self.toxicity_data['Test_Duration_Days']==14)]),
                    len(self.toxicity_data[(self.toxicity_data['Endpoint']=='LC50') & 
                                           (self.toxicity_data['Test_Duration_Days']==28)]),
                    len(self.toxicity_data[self.toxicity_data['Endpoint']=='EC50']),
                    len(self.toxicity_data[self.toxicity_data['Endpoint']=='NOEC']),
                    self.toxicity_data[self.toxicity_data['Endpoint']=='LC50']['Concentration_mg_kg'].median(),
                    self.toxicity_data[self.toxicity_data['Endpoint']=='EC50']['Concentration_mg_kg'].median()
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"\n✓ Exported database to: {filename}")

# ============================================================================
# PART 2: MSAR MODEL IMPLEMENTATION
# ============================================================================

class MSARModel:
    """
    Mean Species Abundance Relationship (MSAR) Model
    Based on Hendriks et al. (2005) framework
    
    Links chemical concentration to population-level effects:
    C → r(C)/r(0) → K(C)/K(0) → MSA(C)
    
    Where:
    - r = intrinsic rate of increase
    - K = carrying capacity
    - MSA = mean species abundance (relative to unexposed)
    """
    
    def __init__(self, toxicity_database):
        self.toxicity_db = toxicity_database
        self.species_parameters = self._initialize_species_parameters()
        
    def _initialize_species_parameters(self):
        """
        Initialize biological parameters for earthworm species
        
        Based on literature:
        - Fründ et al. (2010) - Earthworm population dynamics
        - Holmstrup et al. (2010) - Life history parameters
        """
        params = {
            'Eisenia fetida': {
                'R0': 120,  # Lifetime fecundity (cocoons per lifetime)
                'tau': 90,   # Generation time (days)
                'r0': np.log(120) / 90 * 365,  # Intrinsic rate (per year)
                'body_mass': 0.5,  # Individual body mass (g)
                'ecological_group': 'Epigeic'
            },
            'Eisenia andrei': {
                'R0': 110,
                'tau': 85,
                'r0': np.log(110) / 85 * 365,
                'body_mass': 0.45,
                'ecological_group': 'Epigeic'
            },
            'Lumbricus terrestris': {
                'R0': 80,
                'tau': 365,  # Longer generation time
                'r0': np.log(80) / 365 * 365,
                'body_mass': 5.0,  # Much larger
                'ecological_group': 'Anecic'
            },
            'Lumbricus rubellus': {
                'R0': 100,
                'tau': 180,
                'r0': np.log(100) / 180 * 365,
                'body_mass': 1.0,
                'ecological_group': 'Epigeic'
            },
            'Aporrectodea caliginosa': {
                'R0': 90,
                'tau': 200,
                'r0': np.log(90) / 200 * 365,
                'body_mass': 0.8,
                'ecological_group': 'Endogeic'
            }
        }
        return params
    
    def calculate_fecundity_ratio(self, concentration, LC50, EC50, beta=-0.36, 
                                   qac=3.2, qls=2.5):
        """
        Calculate lifetime fecundity ratio R0(C)/R0(0)
        
        From Hendriks et al. (2005) Equation:
        R0(C)/R0(0) = [1/(1 + (qac*qls*C/LC50)^(1/β))] × [1/(1 + (qac*C/EC50)^(1/β))]
        
        Parameters:
        -----------
        concentration : float
            Exposure concentration (mg/kg soil)
        LC50 : float
            Median lethal concentration
        EC50 : float
            Median effective concentration for reproduction
        beta : float
            Concentration-response slope (typically -0.36)
        qac : float
            Acute-chronic ratio (typically 2.1-4.7, default 3.2)
        qls : float
            Lethal-sublethal ratio (typically 1.3-4.9, default 2.5)
        
        Returns:
        --------
        float : Fecundity ratio (0-1)
        """
        if concentration == 0:
            return 1.0
        
        # Survival component (lethal effects)
        survival_term = 1.0 / (1.0 + (qac * qls * concentration / LC50) ** (1.0 / beta))
        
        # Reproduction component (sublethal effects)
        reproduction_term = 1.0 / (1.0 + (qac * concentration / EC50) ** (1.0 / beta))
        
        fecundity_ratio = survival_term * reproduction_term
        
        return max(fecundity_ratio, 1e-6)  # Avoid exact zero
    
    def calculate_r_ratio(self, concentration, LC50, EC50, R0, beta=-0.36, 
                         qac=3.2, qls=2.5):
        """
        Calculate intrinsic rate of increase ratio r(C)/r(0)
        
        From Hendriks et al. (2005):
        r(C)/r(0) = -[ln(1 + (qac*qls*C/LC50)^(1/β)) + 
                      ln(1 + (qac*C/EC50)^(1/β))] / ln(R0)
        
        Returns:
        --------
        float : Rate ratio (can be negative for decline)
        """
        if concentration == 0:
            return 1.0
        
        term1 = np.log(1.0 + (qac * qls * concentration / LC50) ** (1.0 / beta))
        term2 = np.log(1.0 + (qac * concentration / EC50) ** (1.0 / beta))
        
        r_ratio = -(term1 + term2) / np.log(R0)
        
        return max(r_ratio, -1.0)  # Bound at -1 for extinction
    
    def calculate_K_ratio(self, r_ratio):
        """
        Calculate carrying capacity ratio K(C)/K(0)
        
        Assumption from Hendriks (1996): K changes proportionally with r
        K(C)/K(0) ≈ r(C)/r(0) when qrK ≈ 1
        
        This is valid for moderate contamination levels (C < 0.5 * LC50)
        """
        return max(r_ratio, 0.0)  # Carrying capacity cannot be negative
    
    def calculate_MSA_for_chemical(self, chemical_name, concentration_range, 
                                   species_list=None):
        """
        Calculate Mean Species Abundance for a specific chemical
        
        MSA(C) = (1/N) * Σ[K(C)_i / K(0)_i]
        
        Where N = number of species in community
        
        Parameters:
        -----------
        chemical_name : str
            Name of chemical
        concentration_range : array
            Range of concentrations to evaluate (mg/kg)
        species_list : list
            List of species to include (default: all available)
        
        Returns:
        --------
        DataFrame : Concentration-response relationship with MSA values
        """
        # Get toxicity data for this chemical
        chem_data = self.toxicity_db[
            self.toxicity_db['Chemical_Name'] == chemical_name
        ].copy()
        
        if len(chem_data) == 0:
            print(f"✗ No toxicity data for {chemical_name}")
            return None
        
        if species_list is None:
            species_list = chem_data['Species'].unique()
        
        results = []
        
        for C in concentration_range:
            K_ratios = []
            
            for species in species_list:
                # Get LC50 and EC50 for this species
                species_data = chem_data[chem_data['Species'] == species]
                
                lc50_data = species_data[species_data['Endpoint'] == 'LC50']
                ec50_data = species_data[species_data['Endpoint'] == 'EC50']
                
                if len(lc50_data) == 0 or len(ec50_data) == 0:
                    continue
                
                # Use 14-day LC50 if available, otherwise 28-day
                lc50_14 = lc50_data[lc50_data['Test_Duration_Days'] == 14]
                if len(lc50_14) > 0:
                    LC50 = lc50_14.iloc[0]['Concentration_mg_kg']
                else:
                    LC50 = lc50_data.iloc[0]['Concentration_mg_kg']
                
                EC50 = ec50_data.iloc[0]['Concentration_mg_kg']
                
                # Get species biological parameters
                if species in self.species_parameters:
                    R0 = self.species_parameters[species]['R0']
                else:
                    R0 = 100  # Default
                
                # Calculate r ratio
                r_ratio = self.calculate_r_ratio(C, LC50, EC50, R0)
                
                # Calculate K ratio
                K_ratio = self.calculate_K_ratio(r_ratio)
                
                K_ratios.append(K_ratio)
            
            # Calculate MSA as mean of K ratios
            if len(K_ratios) > 0:
                MSA = np.mean(K_ratios)
                MSA_std = np.std(K_ratios)
            else:
                MSA = 1.0
                MSA_std = 0.0
            
            results.append({
                'Concentration_mg_kg': C,
                'MSA': MSA,
                'MSA_std': MSA_std,
                'N_species': len(K_ratios)
            })
        
        return pd.DataFrame(results)
    
    def calculate_biomass_reduction(self, MSA, baseline_biomass_g_m2=50):
        """
        Convert MSA to earthworm biomass
        
        Assumes biomass is proportional to abundance
        Typical earthworm biomass: 20-100 g/m² in temperate grasslands
        
        Parameters:
        -----------
        MSA : float
            Mean Species Abundance (0-1)
        baseline_biomass_g_m2 : float
            Baseline earthworm biomass in unpolluted soil (g/m²)
        
        Returns:
        --------
        float : Earthworm biomass (g/m²)
        """
        return baseline_biomass_g_m2 * MSA
    
    def plot_MSAR_curve(self, chemical_name, concentration_range=None, 
                       ax=None, show_uncertainty=True):
        """
        Plot MSAR curve for a specific chemical
        
        Shows relationship between concentration and MSA
        """
        if concentration_range is None:
            # Auto-generate concentration range
            chem_data = self.toxicity_db[
                self.toxicity_db['Chemical_Name'] == chemical_name
            ]
            median_lc50 = chem_data[chem_data['Endpoint']=='LC50']['Concentration_mg_kg'].median()
            concentration_range = np.logspace(-2, np.log10(2*median_lc50), 50)
        
        msar_data = self.calculate_MSA_for_chemical(chemical_name, concentration_range)
        
        if msar_data is None:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot MSA curve
        ax.semilogx(msar_data['Concentration_mg_kg'], msar_data['MSA'], 
                   'b-', linewidth=2.5, label='Mean Species Abundance')
        
        if show_uncertainty and 'MSA_std' in msar_data.columns:
            ax.fill_between(
                msar_data['Concentration_mg_kg'],
                msar_data['MSA'] - msar_data['MSA_std'],
                msar_data['MSA'] + msar_data['MSA_std'],
                alpha=0.3, color='blue', label='±1 SD'
            )
        
        # Add reference lines
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='50% reduction (HC5 analog)')
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7,
                  label='80% reduction (severe impact)')
        
        # Find EC50 for MSA (concentration causing 50% reduction)
        if len(msar_data[msar_data['MSA'] <= 0.5]) > 0:
            ec50_msa = msar_data[msar_data['MSA'] <= 0.5].iloc[0]['Concentration_mg_kg']
            ax.axvline(x=ec50_msa, color='red', linestyle=':', alpha=0.5)
            ax.text(ec50_msa, 0.55, f'EC50(MSA)={ec50_msa:.1f}', 
                   rotation=90, va='bottom')
        
        ax.set_xlabel('Soil Concentration (mg/kg dry weight)', fontsize=12)
        ax.set_ylabel('Mean Species Abundance (MSA)', fontsize=12)
        ax.set_title(f'MSAR Curve: {chemical_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        return ax

# ============================================================================
# PART 3: MEURER SOIL STRUCTURE MODEL (from previous implementation)
# ============================================================================

class SoilStructureModel:
    """
    Soil structure dynamics model from Meurer et al. (2020)
    Modified to accept variable bioturbation rates based on earthworm biomass
    """
    
    def __init__(self, baseline_biomass_g_m2=50):
        # Soil parameters (from Table 4, Meurer 2020)
        self.phi = 0.4                    # Total porosity
        self.phi_min = 0.3               # Minimum porosity
        self.ft_mic = 0.8                # Micropore fraction
        self.gamma_s = 2.7               # Particle density [g/cm³]
        self.gamma_r = 1.2               # Root density
        self.fr_c = 0.2                  # Coarse root fraction
        
        # Baseline bioturbation (high scenario from Meurer)
        self.tau_s_baseline = 0.12       # year⁻¹
        
        # Root parameters
        self.Rg = 0.0012                 # Root production [g/cm³/year]
        self.tau_r = 1.0                 # Root turnover rate
        
        # Earthworm parameters
        self.fcasts_mic = 0.8            # Micropore fraction in casts
        self.epsilon_casts = 0.6         # Void ratio of casts
        
        # Baseline earthworm biomass
        self.baseline_biomass_g_m2 = baseline_biomass_g_m2
        
        # Calculate derived parameters
        self._calculate_derived()
    
    def _calculate_derived(self):
        """Calculate derived parameters"""
        self.gamma_b = self.gamma_s * (1 - self.phi)
        self.Br = self.Rg / self.tau_r
    
    def adjust_bioturbation_rate(self, earthworm_biomass_g_m2):
        """
        Adjust bioturbation rate based on earthworm biomass
        
        Assumes linear relationship between biomass and bioturbation
        τ_s(biomass) = τ_s(baseline) × (biomass / baseline_biomass)
        
        Parameters:
        -----------
        earthworm_biomass_g_m2 : float
            Current earthworm biomass (g/m²)
        
        Returns:
        --------
        float : Adjusted bioturbation rate (year⁻¹)
        """
        biomass_ratio = earthworm_biomass_g_m2 / self.baseline_biomass_g_m2
        tau_s_adjusted = self.tau_s_baseline * biomass_ratio
        return max(tau_s_adjusted, 0.001)  # Minimum bioturbation from other sources
    
    def soil_structure_odes(self, y, t, tau_s):
        """
        ODEs for soil structure dynamics with variable bioturbation
        
        Parameters:
        -----------
        y : array
            State vector [φ_mac, φ_s_mes, φ_s_mic]
        t : float
            Time (years)
        tau_s : float
            Bioturbation rate (adjusted for earthworm biomass)
        """
        phi_mac = y[0]
        phi_s_mes = y[1]
        phi_s_mic = y[2]
        
        phi_mat = self.phi_min + phi_s_mes + phi_s_mic
        epsilon = phi_mat / (1 - self.phi)
        
        # Equation 22a: Macroporosity
        root_term = self.fr_c * (self.Br * self.tau_r / self.gamma_r)
        earthworm_term = (epsilon - self.epsilon_casts) * (self.gamma_b * tau_s / self.gamma_s)
        dphi_mac_dt = root_term + earthworm_term
        
        # Equation 22b: Mesoporosity
        if (phi_s_mes + phi_s_mic) > 1e-10:
            frac_mes = phi_s_mes / (phi_s_mes + phi_s_mic)
        else:
            frac_mes = 0.5
        
        root_term = ((1 - self.fr_c) - frac_mes) * (self.Br * self.tau_r / self.gamma_r)
        earthworm_term = ((1 - self.fcasts_mic) * self.epsilon_casts - frac_mes * epsilon) * \
                        (self.gamma_b * tau_s / self.gamma_s)
        dphi_s_mes_dt = root_term + earthworm_term
        
        # Equation 22c: Microporosity
        if (phi_s_mes + phi_s_mic) > 1e-10:
            frac_mic = phi_s_mic / (phi_s_mes + phi_s_mic)
        else:
            frac_mic = 0.5
        
        root_term = -frac_mic * (self.Br * self.tau_r / self.gamma_r)
        earthworm_term = (self.fcasts_mic * self.epsilon_casts - frac_mic * epsilon) * \
                        (self.gamma_b * tau_s / self.gamma_s)
        dphi_s_mic_dt = root_term + earthworm_term
        
        return [dphi_mac_dt, dphi_s_mes_dt, dphi_s_mic_dt]
    
    def simulate_with_varying_biomass(self, years, earthworm_biomass_timeseries, 
                                     initial_conditions):
        """
        Simulate soil structure with time-varying earthworm biomass
        
        Parameters:
        -----------
        years : int
            Simulation duration (years)
        earthworm_biomass_timeseries : array
            Earthworm biomass at each time point (g/m²)
        initial_conditions : dict
            Initial values for φ_mac, φ_s_mes, φ_s_mic
        
        Returns:
        --------
        tuple : (time, solution) where solution contains [φ_mac, φ_mes, φ_mic]
        """
        y0 = [
            initial_conditions['phi_mac'],
            initial_conditions['phi_s_mes'],
            initial_conditions['phi_s_mic']
        ]
        
        t = np.linspace(0, years, len(earthworm_biomass_timeseries))
        solution = np.zeros((len(t), 3))
        solution[0] = y0
        
        # Integrate step by step with updated bioturbation rate
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            biomass = earthworm_biomass_timeseries[i-1]
            tau_s = self.adjust_bioturbation_rate(biomass)
            
            # Single step integration
            t_span = [t[i-1], t[i]]
            sol = odeint(self.soil_structure_odes, solution[i-1], t_span, 
                        args=(tau_s,))
            solution[i] = sol[-1]
        
        return t, solution
    
    def calculate_total_porosities(self, solution):
        """Calculate total porosities from structural components"""
        phi_s_mac = solution[:, 0]
        phi_s_mes = solution[:, 1]
        phi_s_mic = solution[:, 2]
        
        phi_t_mic = self.ft_mic * self.phi_min
        phi_t_mes = (1 - self.ft_mic) * self.phi_min
        
        phi_mic = phi_t_mic + phi_s_mic
        phi_mes = phi_t_mes + phi_s_mes
        phi_mac = phi_s_mac
        phi_mat = phi_mic + phi_mes
        
        return {
            'phi_mic': phi_mic,
            'phi_mes': phi_mes,
            'phi_mac': phi_mac,
            'phi_mat': phi_mat
        }

# ============================================================================
# PART 4: INTEGRATED FRAMEWORK - COUPLING MSAR WITH MEURER
# ============================================================================

class IntegratedChemicalImpactModel:
    """
    Integrated framework coupling:
    1. Chemical exposure (concentration in soil)
    2. MSAR model (population response)
    3. Biomass reduction (from MSA)
    4. Meurer model (soil structure dynamics)
    5. Ecosystem service impacts
    
    Workflow:
    Chemical C → MSA(C) → Biomass(C) → Bioturbation(C) → Soil structure(C)
    """
    
    def __init__(self, toxicity_database):
        self.toxicity_db = toxicity_database
        self.msar = MSARModel(toxicity_database)
        self.soil_model = SoilStructureModel()
        
    def simulate_chemical_impact(self, chemical_name, soil_concentration_mg_kg, 
                                 years=100, exposure_start_year=0):
        """
        Simulate complete impact pathway from chemical exposure to soil structure
        
        Scenarios:
        1. Baseline (no contamination)
        2. Acute contamination (high concentration, short duration)
        3. Chronic contamination (moderate concentration, long duration)
        
        Parameters:
        -----------
        chemical_name : str
            Name of chemical
        soil_concentration_mg_kg : float
            Concentration in soil (mg/kg dry weight)
        years : int
            Simulation duration (years)
        exposure_start_year : int
            When contamination begins
        
        Returns:
        --------
        dict : Complete simulation results with all compartments
        """
        
        print(f"\n{'='*80}")
        print(f"INTEGRATED IMPACT SIMULATION: {chemical_name}")
        print(f"Concentration: {soil_concentration_mg_kg} mg/kg soil")
        print(f"{'='*80}\n")
        
        # Time vector
        t = np.linspace(0, years, 365 * years // 7)  # Weekly time steps
        
        # Step 1: Calculate MSA over time
        print("Step 1: Calculating earthworm population response (MSAR)...")
        
        # Create concentration time series (instant contamination, persistent)
        concentration_timeseries = np.where(
            t >= exposure_start_year, 
            soil_concentration_mg_kg, 
            0
        )
        
        # Calculate MSA for each time point
        MSA_timeseries = np.zeros_like(t)
        for i, C in enumerate(concentration_timeseries):
            if C == 0:
                MSA_timeseries[i] = 1.0
            else:
                msar_result = self.msar.calculate_MSA_for_chemical(
                    chemical_name, 
                    [C]
                )
                if msar_result is not None and len(msar_result) > 0:
                    MSA_timeseries[i] = msar_result.iloc[0]['MSA']
                else:
                    MSA_timeseries[i] = 1.0
        
        print(f"  ✓ MSA reduction: {(1-MSA_timeseries[-1])*100:.1f}%")
        
        # Step 2: Convert MSA to biomass
        print("\nStep 2: Converting MSA to earthworm biomass...")
        baseline_biomass = 50  # g/m²
        biomass_timeseries = baseline_biomass * MSA_timeseries
        print(f"  ✓ Biomass reduction: {baseline_biomass:.1f} → {biomass_timeseries[-1]:.1f} g/m²")
        
        # Step 3: Simulate soil structure with varying biomass
        print("\nStep 3: Simulating soil structure dynamics...")
        initial_conditions = {
            'phi_mac': 0.10,   # Start with healthy soil
            'phi_s_mes': 0.10,
            'phi_s_mic': 0.03
        }
        
        # Baseline scenario (no contamination)
        baseline_biomass_ts = np.ones_like(t) * baseline_biomass
        t_soil, solution_baseline = self.soil_model.simulate_with_varying_biomass(
            years, baseline_biomass_ts, initial_conditions
        )
        porosities_baseline = self.soil_model.calculate_total_porosities(solution_baseline)
        
        # Contaminated scenario
        t_soil, solution_contaminated = self.soil_model.simulate_with_varying_biomass(
            years, biomass_timeseries, initial_conditions
        )
        porosities_contaminated = self.soil_model.calculate_total_porosities(solution_contaminated)
        
        print(f"  ✓ Macroporosity impact:")
        print(f"    Baseline: {porosities_baseline['phi_mac'][-1]:.4f}")
        print(f"    Contaminated: {porosities_contaminated['phi_mac'][-1]:.4f}")
        print(f"    Reduction: {(1 - porosities_contaminated['phi_mac'][-1]/porosities_baseline['phi_mac'][-1])*100:.1f}%")
        
        # Step 4: Calculate ecosystem service impacts
        print("\nStep 4: Assessing ecosystem service impacts...")
        es_impacts = self._calculate_ecosystem_service_impacts(
            porosities_baseline,
            porosities_contaminated
        )
        
        for service, impact in es_impacts.items():
            print(f"  ✓ {service}: {impact['reduction_percent']:.1f}% reduction")
        
        # Compile results
        results = {
            'time_years': t_soil,
            'chemical_name': chemical_name,
            'concentration_mg_kg': soil_concentration_mg_kg,
            'MSA': MSA_timeseries,
            'biomass_g_m2': biomass_timeseries,
            'baseline': {
                'phi_mac': porosities_baseline['phi_mac'],
                'phi_mes': porosities_baseline['phi_mes'],
                'phi_mic': porosities_baseline['phi_mic'],
                'phi_mat': porosities_baseline['phi_mat']
            },
            'contaminated': {
                'phi_mac': porosities_contaminated['phi_mac'],
                'phi_mes': porosities_contaminated['phi_mes'],
                'phi_mic': porosities_contaminated['phi_mic'],
                'phi_mat': porosities_contaminated['phi_mat']
            },
            'ecosystem_services': es_impacts
        }
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"{'='*80}\n")
        
        return results
    
    def _calculate_ecosystem_service_impacts(self, baseline_porosity, 
                                            contaminated_porosity):
        """
        Calculate impacts on ecosystem services
        
        Key services affected by soil structure:
        1. Water infiltration (depends on macroporosity)
        2. Water holding capacity (depends on meso/microporosity)
        3. Soil aeration (depends on total porosity, macropores)
        4. Root penetration (depends on macroporosity, bulk density)
        5. Nutrient cycling (depends on biological activity)
        """
        
        # Water infiltration (proportional to macroporosity)
        infiltration_baseline = baseline_porosity['phi_mac'][-1]
        infiltration_contaminated = contaminated_porosity['phi_mac'][-1]
        infiltration_reduction = (1 - infiltration_contaminated / infiltration_baseline) * 100
        
        # Water holding capacity (depends on meso+micropores)
        whc_baseline = baseline_porosity['phi_mes'][-1] + 0.5 * baseline_porosity['phi_mic'][-1]
        whc_contaminated = contaminated_porosity['phi_mes'][-1] + 0.5 * contaminated_porosity['phi_mic'][-1]
        whc_reduction = (1 - whc_contaminated / whc_baseline) * 100
        
        # Soil aeration (total porosity, weighted toward macropores)
        aeration_baseline = baseline_porosity['phi_mac'][-1] + 0.3 * baseline_porosity['phi_mes'][-1]
        aeration_contaminated = contaminated_porosity['phi_mac'][-1] + 0.3 * contaminated_porosity['phi_mes'][-1]
        aeration_reduction = (1 - aeration_contaminated / aeration_baseline) * 100
        
        return {
            'Water infiltration': {
                'baseline': infiltration_baseline,
                'contaminated': infiltration_contaminated,
                'reduction_percent': infiltration_reduction
            },
            'Water holding capacity': {
                'baseline': whc_baseline,
                'contaminated': whc_contaminated,
                'reduction_percent': whc_reduction
            },
            'Soil aeration': {
                'baseline': aeration_baseline,
                'contaminated': aeration_contaminated,
                'reduction_percent': aeration_reduction
            }
        }
    
    def plot_integrated_results(self, results, save_filename=None):
        """
        Create comprehensive visualization of integrated model results
        
        4-panel figure:
        1. MSA and biomass over time
        2. Macroporosity evolution
        3. Matrix porosity evolution  
        4. Ecosystem service impacts
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        t = results['time_years']
        
        # Panel 1: MSA and Biomass
        ax1 = fig.add_subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        
        ax1.plot(t, results['MSA'], 'b-', linewidth=2.5, label='MSA')
        ax1_twin.plot(t, results['biomass_g_m2'], 'g-', linewidth=2.5, label='Earthworm biomass')
        
        ax1.set_xlabel('Time (years)', fontsize=11)
        ax1.set_ylabel('Mean Species Abundance (MSA)', fontsize=11, color='b')
        ax1_twin.set_ylabel('Earthworm Biomass (g/m²)', fontsize=11, color='g')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        ax1.set_title('(a) Population Response to Contamination', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Add lines
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Panel 2: Macroporosity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, results['baseline']['phi_mac'], 'b-', linewidth=2.5, label='Baseline')
        ax2.plot(t, results['contaminated']['phi_mac'], 'r-', linewidth=2.5, label='Contaminated')
        ax2.fill_between(t, results['baseline']['phi_mac'], results['contaminated']['phi_mac'],
                        alpha=0.3, color='red', label='Impact')
        
        ax2.set_xlabel('Time (years)', fontsize=11)
        ax2.set_ylabel('Macroporosity (cm³/cm³)', fontsize=11)
        ax2.set_title('(b) Macroporosity Degradation', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Matrix Porosity
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t, results['baseline']['phi_mat'], 'b-', linewidth=2.5, label='Baseline')
        ax3.plot(t, results['contaminated']['phi_mat'], 'r-', linewidth=2.5, label='Contaminated')
        ax3.fill_between(t, results['baseline']['phi_mat'], results['contaminated']['phi_mat'],
                        alpha=0.3, color='red')
        
        ax3.set_xlabel('Time (years)', fontsize=11)
        ax3.set_ylabel('Matrix Porosity (cm³/cm³)', fontsize=11)
        ax3.set_title('(c) Matrix Porosity Changes', fontsize=13, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Ecosystem Service Impacts
        ax4 = fig.add_subplot(gs[1, 1])
        
        services = list(results['ecosystem_services'].keys())
        reductions = [results['ecosystem_services'][s]['reduction_percent'] for s in services]
        
        bars = ax4.barh(services, reductions, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax4.set_xlabel('Reduction (%)', fontsize=11)
        ax4.set_title('(d) Ecosystem Service Impacts', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, reductions)):
            ax4.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)
        
        # Overall title
        fig.suptitle(f'Integrated Impact Assessment: {results["chemical_name"]} '
                    f'({results["concentration_mg_kg"]} mg/kg)', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved: {save_filename}")
        
        plt.tight_layout()
        plt.show()
        
        return fig

# ============================================================================
# PART 5: DEMONSTRATION AND SENSITIVITY ANALYSIS
# ============================================================================

def demonstrate_integrated_framework():
    """
    Complete demonstration of the integrated MSAR-Meurer framework
    """
    
    print("\n" + "="*80)
    print("INTEGRATED MSAR-MEURER FRAMEWORK DEMONSTRATION")
    print("Linking Chemical Pollution → Earthworm Populations → Soil Structure")
    print("="*80 + "\n")
    
    # Step 1: Generate synthetic toxicity database
    print("STEP 1: Generating synthetic toxicity database...")
    print("-" * 80)
    db_generator = SyntheticToxicityDatabase(n_chemicals=50, seed=42)
    chemicals = db_generator.generate_chemical_library()
    toxicity_data = db_generator.generate_toxicity_data()
    
    # Export database
    db_generator.export_database('synthetic_earthworm_toxicity.xlsx')
    
    # Step 2: Test MSAR model
    print("\n\nSTEP 2: Testing MSAR model...")
    print("-" * 80)
    msar = MSARModel(toxicity_data)
    
    # Select a test chemical
    test_chemical = 'Imidacloprid'  # Neonicotinoid pesticide
    
    # Generate MSAR curve
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    msar.plot_MSAR_curve(test_chemical, ax=ax1)
    plt.tight_layout()
    plt.savefig('msar_curve_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 3: Run integrated simulation
    print("\n\nSTEP 3: Running integrated simulation...")
    print("-" * 80)
    integrated_model = IntegratedChemicalImpactModel(toxicity_data)
    
    # Scenario 1: Moderate contamination
    results_moderate = integrated_model.simulate_chemical_impact(
        chemical_name=test_chemical,
        soil_concentration_mg_kg=10,  # mg/kg
        years=100,
        exposure_start_year=0
    )
    
    # Visualize results
    integrated_model.plot_integrated_results(
        results_moderate,
        save_filename='integrated_impact_moderate.png'
    )
    
    # Scenario 2: High contamination
    print("\n\n" + "="*80)
    print("SCENARIO 2: High contamination")
    print("="*80)
    results_high = integrated_model.simulate_chemical_impact(
        chemical_name=test_chemical,
        soil_concentration_mg_kg=50,  # mg/kg
        years=100,
        exposure_start_year=0
    )
    
    integrated_model.plot_integrated_results(
        results_high,
        save_filename='integrated_impact_high.png'
    )
    
    # Step 4: Multi-chemical comparison
    print("\n\nSTEP 4: Multi-chemical comparison...")
    print("-" * 80)
    
    test_chemicals = ['Imidacloprid', 'Chlorpyrifos', 'Copper', 'Atrazine']
    test_concentration = 20  # mg/kg
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, chem in enumerate(test_chemicals):
        if chem in toxicity_data['Chemical_Name'].values:
            msar.plot_MSAR_curve(chem, ax=axes[i], show_uncertainty=False)
    
    plt.tight_layout()
    plt.savefig('msar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. synthetic_earthworm_toxicity.xlsx - Toxicity database")
    print("  2. msar_curve_example.png - MSAR curve")
    print("  3. integrated_impact_moderate.png - Moderate contamination scenario")
    print("  4. integrated_impact_high.png - High contamination scenario")
    print("  5. msar_comparison.png - Multi-chemical comparison")
    print("\n" + "="*80 + "\n")

# Run demonstration
if __name__ == "__main__":
    demonstrate_integrated_framework()