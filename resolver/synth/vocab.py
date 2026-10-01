"""Hand-written vocabulary for the synthetic generator: geography, name parts, noise tables.

Everything here is invented or generic (common surnames, trades, street words, real state/region
and city names). There are no real business records and no external lookups.
"""

# ---------------------------------------------------------------- geography
# state code -> (state name, [cities]), plus a zip/PIN/postcode prefix per state.
US_STATES = {
    "CA": ("California", ["Los Angeles", "San Diego", "San Jose", "Fresno", "Sacramento", "Oakland", "Riverside"], "9"),
    "TX": ("Texas", ["Houston", "Dallas", "Austin", "San Antonio", "El Paso", "Lubbock", "Plano"], "7"),
    "NY": ("New York", ["Brooklyn", "Buffalo", "Rochester", "Albany", "Yonkers", "Syracuse"], "1"),
    "FL": ("Florida", ["Miami", "Orlando", "Tampa", "Jacksonville", "Tallahassee", "Gainesville"], "3"),
    "IL": ("Illinois", ["Chicago", "Aurora", "Naperville", "Peoria", "Springfield", "Joliet"], "6"),
    "PA": ("Pennsylvania", ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Scranton", "Lancaster"], "1"),
    "OH": ("Ohio", ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron", "Euclid"], "4"),
    "GA": ("Georgia", ["Atlanta", "Savannah", "Augusta", "Macon", "Athens", "Marietta"], "3"),
    "NC": ("North Carolina", ["Charlotte", "Raleigh", "Durham", "Greensboro", "Wilmington", "Asheville"], "2"),
    "MI": ("Michigan", ["Detroit", "Grand Rapids", "Lansing", "Ann Arbor", "Flint", "Kalamazoo"], "4"),
    "WA": ("Washington", ["Seattle", "Spokane", "Tacoma", "Bellevue", "Olympia", "Everett"], "9"),
    "AZ": ("Arizona", ["Phoenix", "Tucson", "Mesa", "Scottsdale", "Tempe", "Flagstaff"], "8"),
    "KS": ("Kansas", ["Wichita", "Topeka", "Lawrence", "Olathe", "Salina", "Manhattan"], "6"),
    "CO": ("Colorado", ["Denver", "Boulder", "Aurora", "Pueblo", "Fort Collins", "Lakewood"], "8"),
}

IN_STATES = {
    "MH": ("Maharashtra", ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane", "Aurangabad"], "4"),
    "KA": ("Karnataka", ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi"], "5"),
    "TN": ("Tamil Nadu", ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"], "6"),
    "DL": ("Delhi", ["New Delhi", "Dwarka", "Rohini", "Saket", "Karol Bagh"], "1"),
    "GJ": ("Gujarat", ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"], "3"),
    "UP": ("Uttar Pradesh", ["Lucknow", "Kanpur", "Noida", "Agra", "Varanasi", "Prayagraj"], "2"),
    "WB": ("West Bengal", ["Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol"], "7"),
    "TG": ("Telangana", ["Hyderabad", "Warangal", "Secunderabad", "Karimnagar"], "5"),
    "RJ": ("Rajasthan", ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"], "3"),
    "KL": ("Kerala", ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"], "6"),
}
# States whose records sometimes arrive in Devanagari script.
IN_DEVANAGARI = {"MH", "DL", "UP", "RJ", "GJ"}

FR_REGIONS = {
    "IDF": ("Ile-de-France", ["Paris", "Boulogne-Billancourt", "Saint-Denis", "Versailles", "Nanterre"], "75"),
    "ARA": ("Auvergne-Rhone-Alpes", ["Lyon", "Grenoble", "Saint-Etienne", "Clermont-Ferrand", "Annecy"], "69"),
    "PACA": ("Provence-Alpes-Cote d'Azur", ["Marseille", "Nice", "Toulon", "Aix-en-Provence", "Avignon"], "13"),
    "OCC": ("Occitanie", ["Toulouse", "Montpellier", "Nimes", "Perpignan", "Beziers"], "31"),
    "NAQ": ("Nouvelle-Aquitaine", ["Bordeaux", "Limoges", "Poitiers", "La Rochelle", "Pau"], "33"),
    "HDF": ("Hauts-de-France", ["Lille", "Amiens", "Roubaix", "Tourcoing", "Calais"], "59"),
    "GES": ("Grand Est", ["Strasbourg", "Reims", "Metz", "Nancy", "Mulhouse"], "67"),
    "BRE": ("Bretagne", ["Rennes", "Brest", "Quimper", "Lorient", "Vannes"], "35"),
    "PDL": ("Pays de la Loire", ["Nantes", "Angers", "Le Mans", "Saint-Nazaire"], "44"),
}
# Accented display forms used in canonical (S1) French text; records often lose the accents.
FR_ACCENTS = {
    "Ile-de-France": "Île-de-France", "Auvergne-Rhone-Alpes": "Auvergne-Rhône-Alpes",
    "Provence-Alpes-Cote d'Azur": "Provence-Alpes-Côte d'Azur", "Saint-Etienne": "Saint-Étienne",
    "Nimes": "Nîmes", "Beziers": "Béziers",
}

# ---------------------------------------------------------------- streets
US_STREET_NAMES = (
    "Main Oak Maple Washington Lincoln Park Elm Cedar Lake Hill Sunset Euclid Jefferson Madison Franklin "
    "Highland Ridge Spring Center Church Mill Pine Walnut Chestnut River Broadway Jackson Adams Monroe "
    "Willow Cherry Meadow Forest Valley Harbor Grant Sheridan Kingston Prospect Union Liberty Market "
    "Colfax Wilshire Peachtree Magnolia Cypress Juniper Aspen Sycamore Hickory Birch Laurel Dogwood").split()
US_STREET_TYPES = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Boulevard": "Blvd", "Drive": "Dr",
                   "Lane": "Ln", "Court": "Ct", "Place": "Pl", "Parkway": "Pkwy", "Highway": "Hwy",
                   "Terrace": "Ter", "Way": "Way"}
US_UNITS = ["Suite {n}", "Unit {n}", "Ste {n}", "Apt {n}", "Floor {f}", "Bldg {l}"]

IN_STREETS = ["MG Road", "Station Road", "Main Road", "Ring Road", "Link Road", "Nehru Road", "Gandhi Road",
              "Tilak Road", "Market Road", "Temple Street", "Hospital Road", "College Road", "Mall Road",
              "Court Road", "Lake Road", "Bazaar Street", "Church Street", "Industrial Area Road",
              "1st Cross", "2nd Cross", "4th Main", "8th Main", "Sector 12 Road", "Sector 18 Road", "GT Road",
              "SV Road", "LBS Marg", "Bank Street", "Commercial Street", "Avenue Road"]
IN_LOCALITIES = ["Shivaji Nagar", "Gandhi Nagar", "Nehru Colony", "Rajendra Nagar", "Indira Nagar", "Model Town",
                 "Civil Lines", "Ashok Nagar", "Laxmi Nagar", "Sadar Bazar", "Jayanagar", "Koramangala",
                 "Andheri East", "Kothrud", "Navrangpura", "Salt Lake", "Banjara Hills", "T Nagar", "Anna Nagar",
                 "Malviya Nagar", "Vaishali Nagar", "Hazratganj", "Gomti Nagar", "Ballygunge", "Kakkanad",
                 "Bhandup West", "Hadapsar", "Whitefield", "Adyar", "Satellite", "Ram Nagar", "Patel Nagar"]
IN_HOUSE_FMT = ["{h}", "No {h}", "No. {h}", "Plot No {h}", "Door No {h}", "H.No. {h}", "Shop No {h}", "{h}/{s}"]
IN_UNITS = ["Ground Floor", "1st Floor", "2nd Floor", "3rd Floor", "Shop {n}", "Unit {n}"]
IN_BUILDINGS = ["Laxmi Complex", "Sai Plaza", "Krishna Towers", "Ganga Arcade", "Shanti Bhavan", "Om Chambers",
                "Royal Enclave", "City Centre", "Trade Point", "Galaxy Mall"]
IN_LANDMARKS = ["Near SBI ATM", "Opp. Railway Station", "Behind Bus Stand", "Beside HP Petrol Pump",
                "Near Hanuman Mandir", "Opp. Govt Hospital", "Near Clock Tower", "Behind Post Office",
                "Near Bus Depot", "Opp. City Mall", "Near Police Station", "Next to Axis Bank"]

FR_STREET_TYPES = {"Rue": "R.", "Avenue": "Av.", "Boulevard": "Bd", "Place": "Pl.", "Chemin": "Ch.",
                   "Allée": "All.", "Impasse": "Imp.", "Quai": "Qu."}
FR_STREET_NAMES = ["de la Paix", "Victor Hugo", "Jean Jaurès", "de la République", "Pasteur",
                   "du Général de Gaulle", "des Lilas", "Gambetta", "Voltaire", "de Verdun", "Foch", "Carnot",
                   "de la Gare", "du Commerce", "Nationale", "des Écoles", "du Moulin", "Saint-Michel",
                   "de l'Église", "Émile Zola", "Pierre Curie", "des Acacias", "du Port", "Lafayette",
                   "de Strasbourg", "Jules Ferry", "Anatole France", "de Bretagne", "du Marché", "Clemenceau"]
FR_UNITS = ["Bât. {l}", "Étage {f}", "Lot {n}", "BP {n}"]

# ---------------------------------------------------------------- name parts
US_SURNAMES = (
    "Johnson Smith Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Wilson Anderson "
    "Thomas Taylor Moore Jackson Martin Lee Thompson White Harris Clark Lewis Robinson Walker Young Allen King "
    "Wright Scott Torres Nguyen Hill Flores Green Adams Nelson Baker Hall Rivera Campbell Mitchell Carter Roberts "
    "Phillips Evans Turner Parker Collins Edwards Stewart Morris Murphy Cook Rogers Morgan Peterson Cooper Reed "
    "Bailey Bell Gomez Kelly Howard Ward Cox Diaz Richardson Wood Watson Brooks Bennett Gray James Reyes Hughes "
    "Price Myers Long Foster Sanders Ross Morales Powell Sullivan Russell Ortiz Jenkins Perry Butler Barnes "
    "Fisher Henderson Coleman Simmons Patterson Jordan Reynolds Hamilton Graham Kim Gonzales Alexander Ramos "
    "Wallace Griffin West Cole Hayes Chavez Gibson Bryant Ellis Stevens Murray Ford Marshall Owens McDonald "
    "Harrison Kennedy Wells Yazzie Kowalski Okafor Lindqvist Castellano Brennan Delgado Fitzgerald").split()
US_ADJ = ("Blue Golden Silver Green Red Summit Pioneer Liberty Eagle Coastal Northern Southern Western Eastern "
          "Prime Premier Elite Royal Bright Rapid Grand Evergreen Heritage Frontier Keystone Lakeside Riverside "
          "Mountain Prairie Sunrise Sunset Crystal Diamond Iron Copper Cedar Oakwood Maple Harbor Beacon Atlas "
          "Apex Vertex Patriot Granite Redwood Bluebird Falcon Phoenix Horizon Unity Allied United Precision").split()
US_NOUNS = ("Ridge Valley Creek Point Peak Bay Grove Hills Springs Gate Bridge Crest Field Stone Star Path Way "
            "Harbor Lake Mesa Canyon Prairie Meadow Brook Haven Rock Pine Oak Empire Program Line Works").split()
US_INDUSTRIES = (
    "Plumbing|Dental|Construction|Auto Repair|Landscaping|Roofing|Insurance|Realty|Bakery|Pharmacy|Consulting|"
    "Logistics|Electric|Hardware|Fitness|Cleaning Services|Law Group|Medical Center|Pediatrics|Veterinary Clinic|"
    "Marketing|Software|Printing|Catering|Florist|Furniture|Tire Center|Painting|Accounting|Physical Therapy|"
    "Chiropractic|Daycare|Salon|Barber Shop|Diner|Pizza|Coffee|Brewing|Motors|Storage|Moving|HVAC|Pest Control|"
    "Optometry|Surgical Care|Orthodontics|Engineering|Architecture|Staffing|Security|Transport|Trucking|Machine Shop|"
    "Welding|Glass|Flooring|Tile|Pool Service|Appliance Repair|Car Wash|Laundromat|Tailoring|Jewelers").split("|")
US_LEGAL = ["LLC", "Inc", "Corp", "Co", "Ltd", "LP", "PLLC", "PC", "Corporation", "Incorporated", "Company"]

IN_SURNAMES = ("Sharma Gupta Patel Singh Kumar Verma Agarwal Jain Mehta Shah Reddy Rao Nair Iyer Menon Pillai "
               "Das Ghosh Banerjee Chatterjee Mukherjee Bose Sen Joshi Kulkarni Deshpande Patil Pawar Jadhav "
               "Chauhan Yadav Mishra Tiwari Pandey Dubey Saxena Srivastava Malhotra Kapoor Khanna Bhatia Arora "
               "Chopra Sethi Bansal Goyal Mittal Garg Singhal Khandelwal Rathore Shekhawat Naidu Krishnan "
               "Subramanian Venkatesh Hegde Shetty Kamath Bhat Desai Trivedi Thakkar Modi Parikh Chaudhary").split()
IN_PREFIX = ("Shree Sri Shri Om Sai Jai Maa Ganesh Krishna Balaji Laxmi Durga Shiv Hanuman Ganga Annapurna Vishnu "
             "Mahalakshmi Murugan Ambika Saraswati Tirupati Vinayak Siddhi Gayatri Bharat Hind Rashtriya").split()
IN_WORDS = ("Sunrise Royal Star Galaxy Global National Supreme Perfect Classic Modern Nova Unique Prime Vision "
            "Apex Aditya Akash Amrit Anand Arihant Kaveri Narmada Himalaya Sahyadri Vindhya Konark Kohinoor "
            "Navratna Swastik Kalpataru Pragati Unnati Samruddhi Pratham Shubh Mangal Kamdhenu Neel Kesari").split()
IN_INDUSTRIES = (
    "Traders|Enterprises|Textiles|Electricals|Hardware|Pharma|Infotech|Technology|Engineering Works|Steel|"
    "Agencies|Distributors|Jewellers|Sweets|Caterers|Motors|Auto Parts|Builders|Constructions|Chemicals|Plastics|"
    "Polymers|Garments|Handlooms|Furniture|Opticals|Medicals|Dairy|Foods|Spices|Rice Mill|Oil Mill|Packaging|"
    "Printers|Travels|Logistics|Cargo Movers|Tiles|Ceramics|Paints|Sanitary Stores|Mobile Point|Computers|"
    "Book Depot|Stationers|Tutorials|Diagnostics|Clinic|Hospital|Dental Care|Fabricators|Industries").split("|")
IN_LEGAL = ["Pvt Ltd", "Private Limited", "Ltd", "LLP", "Limited"]

FR_SURNAMES = ("Martin Bernard Dubois Thomas Robert Richard Petit Durand Leroy Moreau Simon Laurent Lefebvre Michel "
               "Garcia David Bertrand Roux Vincent Fournier Morel Girard Andre Lefevre Mercier Dupont Lambert "
               "Bonnet Francois Martinez Legrand Garnier Faure Rousseau Blanc Guerin Muller Henry Roussel Nicolas "
               "Perrin Morin Mathieu Clement Gauthier Dumont Lopez Fontaine Chevalier Robin Masson Sanchez Gerard "
               "Nguyen Boyer Denis Lemaire Duval Joly Gautier Roger Roche Roy Noel Meyer Lucas Meunier Jean Perez "
               "Marchand Dufour Blanchard Marie Barbier Brun Dumas Brunet Schmitt Leroux Colin Fernandez Renaud "
               "Arnaud Rolland Caron Aubert Giraud Leclerc Vidal Bourgeois Renard Lemoine Picard Gaillard Philippe").split()
FR_TRADES = ["Boulangerie", "Pâtisserie", "Garage", "Pharmacie", "Cabinet", "Transports", "Menuiserie", "Plomberie",
             "Électricité", "Boucherie", "Fromagerie", "Librairie", "Coiffure", "Optique", "Imprimerie", "Brasserie",
             "Restaurant", "Hôtel", "Agence Immobilière", "Bâtiment", "Peinture", "Carrosserie", "Fleuriste",
             "Taxi", "Ambulances", "Charpente", "Maçonnerie", "Informatique", "Auto-École", "Traiteur"]
FR_SUFFIX_WORDS = ["Conseil", "Services", "Distribution", "Immobilier", "Rénovation", "Habitat", "Création",
                   "Solutions", "Ingénierie", "Négoce", "Énergie", "Industrie", "Logistique", "Patrimoine"]
FR_PLACES = ["du Centre", "de la Gare", "du Port", "des Halles", "du Marché", "de la Place", "du Lac", "des Alpes",
             "du Midi", "de l'Ouest", "du Nord", "de Provence", "de Bretagne", "Saint-Martin", "du Parc"]
FR_LEGAL = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI"]

# Generic descriptor words that "neighbour" fake records add (the adaptive-vocabulary signal, E-T2).
GENERIC_WORDS = {
    "US": ["Holdings", "Group", "Enterprises", "Solutions", "Services", "International", "Ventures", "Partners",
           "Associates", "Management", "Industries", "Systems", "Global", "Properties", "Investments"],
    "India": ["Enterprises", "Industries", "Traders", "Associates", "Solutions", "Infra", "Ventures", "Group",
              "Exports", "Agencies", "International", "Projects", "Services", "Holdings"],
    "France": ["Développement", "Participations", "Holding", "Groupe", "Conseil", "Gestion", "Investissements",
               "Services", "Patrimoine", "International", "Partenaires", "Finance"],
}

# Invented chain brands (multi-branch businesses: same name, different addresses).
CHAINS = {
    "US": ["Metro Mart", "QuickFix Auto", "Green Leaf Pharmacy", "Urban Cuts", "Sunny Side Diner",
           "Prime Storage", "Bright Smile Dental", "Rapid Lube", "Corner Grocer", "FitZone Gym"],
    "India": ["Annapurna Sweets", "Sai Medicals", "Balaji Mobile Point", "Royal Opticals", "Krishna Dairy",
              "Star Computers", "Ganesh Hardware", "Om Sai Travels", "Nova Diagnostics", "Laxmi Jewellers"],
    "France": ["Boulangerie Le Fournil", "Pharmacie Centrale", "Garage du Rond-Point", "Optique Lumière",
               "Coiffure Élégance", "Fromagerie Saint-Jean", "Auto-École Liberté", "Brasserie du Commerce"],
}

# ---------------------------------------------------------------- noise tables
LEGAL_VARIANTS = {
    "LLC": ["LLC", "L.L.C.", "Llc", "LLC."], "Inc": ["Inc", "Inc.", "Incorporated", "INC"],
    "Incorporated": ["Incorporated", "Inc", "Inc."], "Corp": ["Corp", "Corp.", "Corporation"],
    "Corporation": ["Corporation", "Corp", "Corp."], "Co": ["Co", "Co.", "Company"],
    "Company": ["Company", "Co", "Co."], "Ltd": ["Ltd", "Ltd.", "Limited"], "Limited": ["Limited", "Ltd", "Ltd."],
    "LP": ["LP", "L.P."], "PLLC": ["PLLC", "P.L.L.C."], "PC": ["PC", "P.C."],
    "Pvt Ltd": ["Pvt Ltd", "Pvt. Ltd.", "Private Limited", "Pvt. Ltd", "P Ltd", "(P) Ltd"],
    "Private Limited": ["Private Limited", "Pvt Ltd", "Pvt. Ltd.", "Pvt Limited"], "LLP": ["LLP", "L.L.P."],
    "SARL": ["SARL", "S.A.R.L.", "Sarl"], "SAS": ["SAS", "S.A.S.", "Sas"], "SASU": ["SASU", "S.A.S.U."],
    "EURL": ["EURL", "E.U.R.L."], "SA": ["SA", "S.A."], "SCI": ["SCI", "S.C.I."],
}
WORD_ABBREV = {
    "Associates": "Assoc.", "Brothers": "Bros", "International": "Intl", "Services": "Svcs",
    "Technologies": "Tech", "Technology": "Tech", "Manufacturing": "Mfg", "Center": "Ctr", "Centre": "Ctr",
    "Enterprises": "Ent.", "Industries": "Inds", "Engineering": "Engg", "Electricals": "Elec.",
    "Distributors": "Distr.", "Construction": "Constr.", "Constructions": "Constr.", "Management": "Mgmt",
    "Medical": "Med.", "Pharmacy": "Pharm.", "Insurance": "Ins.", "Realty": "Rlty", "Mountain": "Mtn",
    "Saint": "St", "Société": "Sté", "Établissements": "Ets", "Compagnie": "Cie", "Transports": "Transp.",
    "Informatique": "Info", "Agencies": "Agen.", "Traders": "Trdrs", "Textiles": "Tex",
}
