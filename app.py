# app.py
# --------------------------------------------------------------------------------
# Agri-AI Recommendation System (Dashboard, Recommender, Auth)
# --------------------------------------------------------------------------------

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import json
import random

# --- ML Modules ---
import crop_recommender
import fertilizer_recommender

# --- CONFIGURATION ---
app = Flask(__name__)
# WARNING: Change this key immediately for production use!
app.config['SECRET_KEY'] = 'a_very_secret_and_long_key_for_security_42'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# --- Static Constants ---
SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey', 'Alluvial']
DEFAULT_STATE = 'Maharashtra'
DEFAULT_CITY = 'Pune'

# --- EXTENDED LOCATION DATA (Re-integrated for runnable code) ---
# location_data.py

LOCATION_DATA = {
    "Andhra Pradesh": [
        "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna",
        "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam",
        "Vizianagaram", "West Godavari", "YSR Kadapa"
    ],
    "Arunachal Pradesh": [
        "Tawang", "West Kameng", "East Kameng", "Papum Pare", "Kurung Kumey",
        "Kra Daadi", "Lower Subansiri", "Upper Subansiri", "West Siang",
        "East Siang", "Siang", "Upper Siang", "Lower Siang", "Lower Dibang Valley",
        "Dibang Valley", "Anjaw", "Lohit", "Namsai", "Changlang", "Tirap",
        "Longding"
    ],
    "Assam": [
        "Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo",
        "Chirang", "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Goalpara",
        "Golaghat", "Hailakandi", "Hojai", "Jorhat", "Kamrup", "Kamrup Metropolitan",
        "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon",
        "Nagaon", "Nalbari", "Sivasagar", "Sonitpur", "South Salmara-Mankachar",
        "Tinsukia", "Udalguri", "West Karbi Anglong"
    ],
    "Bihar": [
        "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur",
        "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj",
        "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj",
        "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur",
        "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur",
        "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan", "Supaul", "Vaishali",
        "West Champaran"
    ],
    "Chhattisgarh": [
        "Balod", "Baloda Bazar", "Balrampur", "Bastar", "Bemetara", "Bijapur",
        "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband", "Janjgir-Champa",
        "Jashpur", "Kabirdham", "Kanker", "Kondagaon", "Korba", "Koriya", "Mahasamund",
        "Mungeli", "Narayanpur", "Raigarh", "Raipur", "Rajnandgaon", "Sukma", "Surajpur", "Surguja"
    ],
    "Gujarat": [
        "Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch",
        "Bhavnagar", "Botad", "Chhota Udepur", "Dahod", "Dang", "Devbhoomi Dwarka",
        "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch",
        "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal",
        "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar",
        "Tapi", "Vadodara", "Valsad"
    ],
    "Haryana": [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad",
        "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal",
        "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula", "Panipat",
        "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"
    ],
    "Karnataka": [
        "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban",
        "Bidar", "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga",
        "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri",
        "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur",
        "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"
    ],
    "Maharashtra": [
        "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara",
        "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli",
        "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban",
        "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar",
        "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg",
        "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
    ],
    "Odisha": [
        "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh",
        "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur",
        "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Kendujhar",
        "Khordha", "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh",
        "Nuapada", "Puri", "Rayagada", "Sambalpur", "Sonepur", "Sundargarh"
    ],
    "Punjab": [
        "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib",
        "Fazilka", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala",
        "Ludhiana", "Mansa", "Moga", "Muktsar", "Nawanshahr", "Pathankot",
        "Patiala", "Rupnagar", "Sangrur", "SAS Nagar", "Tarn Taran"
    ],
    "Rajasthan": [
        "Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur",
        "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa",
        "Dholpur", "Dungarpur", "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore",
        "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Kota", "Nagaur",
        "Pali", "Pratapgarh", "Rajsamand", "Sawai Madhopur", "Sikar", "Sirohi",
        "Sri Ganganagar", "Tonk", "Udaipur"
    ],
    "Tamil Nadu": [
        "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
        "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram",
        "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Nagapattinam",
        "Namakkal", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet",
        "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi",
        "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur",
        "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"
    ],
    "Uttar Pradesh": [
        "Agra", "Aligarh", "Allahabad", "Ambedkar Nagar", "Amethi", "Amroha",
        "Auraiya", "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur",
        "Banda", "Barabanki", "Bareilly", "Basti", "Bhadohi", "Bijnor",
        "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah",
        "Etawah", "Faizabad", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddh Nagar",
        "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur",
        "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj",
        "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kheri",
        "Kushinagar", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri",
        "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar",
        "Pilibhit", "Pratapgarh", "Rae Bareli", "Rampur", "Saharanpur",
        "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", "Shamli", "Shravasti",
        "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"
    ]
}


# --- ML Model Load ---
CROP_DATA_PATH = 'Crop_data.csv'
FERT_DATA_PATH = 'Fertilizer_data.csv'

model_load_status = {'crop_loaded': False, 'fert_loaded': False, 'error': None}

print("\n--- Loading ML Models on Startup ---")
try:
    if crop_recommender.load_and_train_crop_model(CROP_DATA_PATH):
        model_load_status['crop_loaded'] = True
    if fertilizer_recommender.load_and_train_fertilizer_model(FERT_DATA_PATH):
        model_load_status['fert_loaded'] = True
    print("✅ Models Loaded Successfully!")
except Exception as e:
    model_load_status['error'] = f"Model Loading Error: {e}"
    print(model_load_status['error'])


# --------------------------------------------------------------------------------
# HELPER FUNCTION: Generate Simulated Dynamic Data
# --------------------------------------------------------------------------------
def fetch_dynamic_content(city, state):
    """Simulate weather, news, and growth data based on location."""
    
    # --- 1. Weather Data ---
    temp = random.randint(18, 35)
    condition = random.choice(["Clear Sky", "Partly Cloudy", "Hazy", "Chance of Showers"])

    weather = {
        "location": f"{city}, {state}",
        "temp": f"{temp}°C",
        "condition": condition,
        "details": f"Humidity: {random.randint(50, 90)}%, Wind: {random.randint(5, 15)} km/h"
    }

    # --- 2. News Headlines ---
    news = [
        f"Local authorities in {city} warn farmers about water table depletion.",
        "State government announces new subsidy scheme for fertilizers.",
        f"Agriculture trends improving across {state} due to early rains."
    ]

    # --- 3. Farming Growth Ratios ---
    crops = random.sample(['Wheat', 'Rice', 'Sugarcane', 'Cotton', 'Maize', 'Groundnut'], 3)
    growth = []
    
    for crop in crops:
        ratio_val = round(random.uniform(-4.5, 6.5), 1)
        growth.append({
            "crop": crop, 
            "ratio": ratio_val,
            "status": "up" if ratio_val >= 0 else "down" # Correct logic
        })

    return {"weather": weather, "news": news, "growth": growth}


# --------------------------------------------------------------------------------
# DATABASE MODELS & AUTH
# --------------------------------------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# --------------------------------------------------------------------------------
# AUTH ROUTES (Standard Flask-Login setup)
# --------------------------------------------------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


# --------------------------------------------------------------------------------
# STATIC PAGES
# --------------------------------------------------------------------------------
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Your contact form processing logic
        print(f"📩 Contact form submitted: {request.form}")
        flash('Your message has been received!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')


# --------------------------------------------------------------------------------
# 7. HOME DASHBOARD (Dashboard Landing Page)
# --------------------------------------------------------------------------------
@app.route('/')
@login_required
def home():
    """Renders the main dashboard page with dynamic features."""
    
    # Get user's selected location from the URL parameters
    state = request.args.get('state', DEFAULT_STATE)
    city = request.args.get('city', DEFAULT_CITY)
    
    # Fallback: If a state is selected but no city (e.g., initial state selection), default to the first city
    if state in LOCATION_DATA and not city:
        city = LOCATION_DATA[state][0]
    
    data = fetch_dynamic_content(city, state)
    
    return render_template('home_dashboard.html',
                           current_state=state,
                           current_city=city,
                           location_data=LOCATION_DATA,
                           weather=data['weather'],
                           news_headlines=data['news'],
                           growth_ratios=data['growth'])


# --------------------------------------------------------------------------------
# 8. RECOMMENDER PAGE (ML Recommendation)
# --------------------------------------------------------------------------------
@app.route('/recommender', methods=['GET', 'POST'])
@login_required
def recommender_page(): # This function name matches the url_for('recommender_page') link in the dashboard
    recommendation = None

    if not model_load_status['crop_loaded'] or not model_load_status['fert_loaded']:
        flash("ML Models not loaded properly. Please check console.", 'danger')
        return render_template('index.html', soil_types=SOIL_TYPES)

    if request.method == 'POST':
        try:
            # Data validation and casting
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            ph = float(request.form['pH'])
            temp = float(request.form['temp'])
            hum = float(request.form['hum'])
            rain = float(request.form['rain'])
            soil = request.form['soil_type']

            # Run ML Models
            crop, conf = crop_recommender.recommend_crop(N, P, K, temp, hum, ph, rain)
            fert = fertilizer_recommender.recommend_fertilizer(
                N=N, P=P, K=K, temp=temp, humidity=hum, ph=ph, soil_type=soil
            )

            # Package result
            recommendation = {
                'crop': crop.upper(),
                'fertilizer': fert.upper(),
                'confidence': f"{conf:.2f}%",
                'inputs': {'N': N, 'P': P, 'K': K, 'pH': ph, 'Temp': temp, 'Hum': hum, 'Rain': rain, 'Soil': soil}
            }
            flash("✅ Recommendation generated successfully!", 'success')

        except Exception as e:
            flash(f"Error processing request (check your input values): {e}", 'danger')

    return render_template('index.html', recommendation=recommendation, soil_types=SOIL_TYPES)


# --------------------------------------------------------------------------------
# 9. RUN APP
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("🚀 Flask app running on http://127.0.0.1:5000")
    app.run(debug=True)