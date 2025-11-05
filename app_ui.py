from flask import Flask, request, send_file, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import os, jwt, pandas as pd, json
import networkx as nx
import spacy, time
from transformers import pipeline
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import sqlite3
import csv
import datetime

# ===============================================================
# Neo4j Integration
# ===============================================================
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
    print("✅ Neo4j driver available")
except ImportError:
    NEO4J_AVAILABLE = False
    print("❌ Neo4j driver not available. Install with: pip install neo4j")

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def test_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Neo4j Connection Successful' AS message")
                return result.single()["message"]
        except Exception as e:
            return f"Connection failed: {str(e)}"
    
    def add_triple(self, entity1, relation, entity2):
        """Add a triple to Neo4j database"""
        with self.driver.session() as session:
            query = (
                "MERGE (a:Entity {name: $entity1}) "
                "MERGE (b:Entity {name: $entity2}) "
                "MERGE (a)-[r:" + relation + "]->(b)"
            )
            session.run(query, entity1=entity1, entity2=entity2)
    
    def add_triples_batch(self, triples):
        """Add multiple triples to Neo4j in batch"""
        if not triples:
            return 0
            
        added_count = 0
        with self.driver.session() as session:
            for triple in triples:
                try:
                    if isinstance(triple, dict):
                        e1, rel, e2 = triple["entity1"], triple["relation"], triple["entity2"]
                    else:
                        e1, rel, e2 = triple
                    
                    # Clean and validate
                    e1 = str(e1).strip()
                    rel = str(rel).strip().upper().replace(' ', '_')
                    e2 = str(e2).strip()
                    
                    if not e1 or not rel or not e2:
                        continue
                        
                    query = (
                        "MERGE (a:Entity {name: $entity1}) "
                        "MERGE (b:Entity {name: $entity2}) "
                        "MERGE (a)-[r:" + rel + "]->(b) "
                        "RETURN a.name, type(r), b.name"
                    )
                    result = session.run(query, entity1=e1, entity2=e2)
                    
                    # Check if the triple was actually created
                    record = result.single()
                    if record:
                        added_count += 1
                        
                except Exception as e:
                    print(f"Error adding triple to Neo4j: {e}")
                    continue
                    
        return added_count
    
    def query_relations(self, entity_name):
        """Query all relations for a given entity"""
        with self.driver.session() as session:
            query = (
                "MATCH (a:Entity {name: $entity_name})-[r]->(b) "
                "RETURN a.name, type(r), b.name"
            )
            result = session.run(query, entity_name=entity_name)
            return [(record["a.name"], record["type(r)"], record["b.name"]) for record in result]
    
    def get_all_triples(self, limit=100):
        """Get all triples from Neo4j"""
        with self.driver.session() as session:
            query = (
                "MATCH (a)-[r]->(b) "
                "RETURN a.name, type(r), b.name "
                "LIMIT $limit"
            )
            result = session.run(query, limit=limit)
            return [(record["a.name"], record["type(r)"], record["b.name"]) for record in result]
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            return "Database cleared"
    
    def get_graph_stats(self):
        """Get Neo4j graph statistics"""
        with self.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = node_result.single()["node_count"]
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = rel_result.single()["rel_count"]
            
            # Count entity types
            entity_types = session.run("""
                MATCH (n:Entity)
                RETURN count(n) as entity_count
            """)
            entity_count = entity_types.single()["entity_count"]
            
            return {
                "nodes": node_count,
                "relationships": rel_count,
                "entities": entity_count
            }

# Initialize Neo4j connection (modify URI and credentials as needed)
def init_neo4j_connection():
    """Initialize Neo4j connection with fallback"""
    if not NEO4J_AVAILABLE:
        return None
        
    try:
        # Default configuration - modify these for your setup
        NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://48b805a3.databases.neo4j.io")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "PcLHx06p5zUx23gOe1OU97ffkXUTIxV2Rjr1cMHRT2M")
        
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Test connection
        test_result = neo4j_conn.test_connection()
        print(f"✅ Neo4j: {test_result}")
        
        return neo4j_conn
        
    except Exception as e:
        print(f"❌ Neo4j initialization failed: {e}")
        return None

# Initialize Neo4j
neo4j_conn = init_neo4j_connection()

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ===============================================================
# Database Models (unchanged)
# ===============================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'

class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    language = db.Column(db.String(50))
    interests = db.Column(db.String(200))

with app.app_context():
    db.create_all()
    for user in User.query.all():
        if not user.role:
            user.role = 'user'
    db.session.commit()

# ===============================================================
# Configuration & Initialization (unchanged)
# ===============================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"txt", "csv", "json", "xml"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================================================
# Authentication & Authorization (unchanged)
# ===============================================================

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token missing"}), 403
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({"error": "Invalid user"}), 403
        except Exception as e:
            return jsonify({"error": f"Invalid/expired token: {str(e)}"}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(current_user, *args, **kwargs):
        if current_user.role != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# ===============================================================
# NLP Models Initialization (unchanged)
# ===============================================================

print("Loading NLP models...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded")
except Exception as e:
    print(f"spaCy model error: {e}")
    nlp = None

try:
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else -1
    re_pipeline = pipeline("text2text-generation", model="Babelscape/rebel-large", device=device)
    print(f"REBEL model loaded (GPU: {use_gpu})")
except Exception as e:
    print(f"REBEL model not available: {e}")
    re_pipeline = None

# ===============================================================
# Semantic Search Engine (unchanged)
# ===============================================================

class SemanticSearchEngine:
    def __init__(self):
        self.model = None
        self.node_embeddings = None
        self.nodes = None
        self.graph = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            print("Initializing semantic search model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Semantic search model loaded")
        except Exception as e:
            print(f"Failed to load semantic search model: {e}")
            self.model = None

    def build_index(self, graph):
        """Build semantic index from graph nodes"""
        if self.model is None:
            print("Cannot build index: model not loaded")
            return 0

        self.graph = graph
        self.nodes = list(graph.nodes())

        print(f"Building semantic index for {len(self.nodes)} nodes...")

        if not self.nodes:
            print("No nodes to index")
            self.node_embeddings = np.array([])
            return 0

        try:
            self.node_embeddings = self.model.encode(
                self.nodes,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            print(f"Semantic index built with {len(self.nodes)} nodes")
            return len(self.nodes)

        except Exception as e:
            print(f"Error building semantic index: {e}")
            self.node_embeddings = None
            return 0

    def search_nodes(self, query, top_k=5, min_score=0.3):
        """Search for most similar nodes"""
        if (self.model is None or
            self.nodes is None or
            self.node_embeddings is None or
            len(self.node_embeddings) == 0):
            return []

        if not query or not query.strip():
            return []

        try:
            query_embedding = self.model.encode([query.strip()])
            similarities = cosine_similarity(query_embedding, self.node_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= min_score:
                    node_name = self.nodes[idx]
                    degree = self.graph.degree(node_name) if self.graph and node_name in self.graph else 0

                    results.append({
                        'node': node_name,
                        'score': float(score),
                        'degree': degree,
                        'rank': len(results) + 1
                    })

            return results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def search_to_subgraph(self, query, top_k=3, radius=1):
        """Generate subgraph from search results"""
        top_nodes = self.search_nodes(query, top_k, min_score=0.1)

        if not top_nodes or not self.graph:
            return None, []

        node_names = [result['node'] for result in top_nodes]

        subgraphs = []
        for node in node_names:
            if node in self.graph:
                try:
                    ego_graph = nx.ego_graph(self.graph, node, radius=radius, undirected=True)
                    subgraphs.append(ego_graph)
                except:
                    continue

        if subgraphs:
            union_subgraph = nx.compose_all(subgraphs)
            return union_subgraph, top_nodes
        else:
            return None, top_nodes

search_engine = SemanticSearchEngine()

# ===============================================================
# Utility Functions (modified to include Neo4j)
# ===============================================================

def run_spacy_ner(text):
    if nlp is None:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def run_dependency_rel(text):
    if nlp is None:
        return []
    doc = nlp(text)
    triples = []
    ent_map = {}
    ent_label = {}

    for ent in doc.ents:
        for token in ent:
            ent_map[token.i] = ent.text
            ent_label[token.i] = ent.label_

    for token in doc:
        if token.pos_ == "VERB":
            subjects = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.children if w.dep_ in ("dobj", "attr")]
            preps = [w for w in token.children if w.dep_ == "prep"]

            for subj in subjects:
                for obj in objects:
                    subj_text = ent_map.get(subj.i, subj.text)
                    obj_text = ent_map.get(obj.i, obj.text)
                    triples.append((subj_text, token.lemma_, obj_text))

            for subj in subjects:
                for prep in preps:
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":
                            subj_text = ent_map.get(subj.i, subj.text)
                            obj_text = ent_map.get(pobj.i, pobj.text)
                            relation = token.lemma_ + " " + prep.text
                            triples.append((subj_text, relation, obj_text))

    for ent in doc.ents:
        if ent.label_ in ("ORG", "FAC"):
            for tok in ent.root.subtree:
                if tok.i in ent_label and ent_label[tok.i] == "GPE":
                    triples.append((ent.text, "located_in", ent_map[tok.i]))

    return triples

def run_rebel(text, timeout=20):
    if not re_pipeline:
        return []
    try:
        start = time.time()
        out = re_pipeline(text, max_length=256, truncation=True)
        if time.time() - start > timeout:
            return []
        raw = out[0]['generated_text']
        triples = []
        for line in raw.splitlines():
            if '|||' in line:
                parts = [p.strip() for p in line.split('|||')]
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
        return triples
    except:
        return []

def save_triples_to_kb(triples):
    """Save extracted triples to the knowledge base database"""
    if not triples:
        return 0
    
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()
    
    saved_count = 0
    for triple in triples:
        try:
            if isinstance(triple, dict) and "entity1" in triple and "relation" in triple and "entity2" in triple:
                e1, rel, e2 = triple["entity1"], triple["relation"], triple["entity2"]
                
                cursor.execute(
                    "SELECT id FROM triples WHERE entity1=? AND relation=? AND entity2=?",
                    (e1, rel, e2)
                )
                existing = cursor.fetchone()
                
                if not existing:
                    cursor.execute(
                        "INSERT INTO triples (entity1, relation, entity2) VALUES (?, ?, ?)",
                        (e1, rel, e2)
                    )
                    saved_count += 1
        except Exception as e:
            print(f"Error saving triple to KB: {e}")
            continue
    
    conn.commit()
    conn.close()
    print(f"Saved {saved_count} new triples to knowledge base")
    return saved_count

def save_triples_to_neo4j(triples):
    """Save extracted triples to Neo4j graph database"""
    if not triples or not neo4j_conn:
        return 0
    
    try:
        neo4j_count = neo4j_conn.add_triples_batch(triples)
        print(f"Saved {neo4j_count} triples to Neo4j")
        return neo4j_count
    except Exception as e:
        print(f"Error saving to Neo4j: {e}")
        return 0

# ===============================================================
# Knowledge Base Database Initialization (unchanged)
# ===============================================================

def init_kb_database():
    """Initialize the knowledge base database"""
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity1 TEXT,
            relation TEXT,
            entity2 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Knowledge Base database initialized")

def init_feedback_file():
    """Initialize feedback CSV file"""
    if not os.path.exists('feedback.csv'):
        df = pd.DataFrame(columns=['timestamp', 'user_id', 'username', 'query', 'rating', 'comments'])
        df.to_csv('feedback.csv', index=False)
        print("Feedback file initialized")

init_kb_database()
init_feedback_file()

# ===============================================================
# Core API Routes (unchanged)
# ===============================================================

@app.route("/health", methods=["GET"])
def health_check():
    neo4j_status = "available" if neo4j_conn else "unavailable"
    return jsonify({
        "status": "healthy",
        "message": "Flask server is running",
        "neo4j": neo4j_status,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route("/test", methods=["GET"])
def test_endpoint():
    return jsonify({"message": "Test successful - server is working!"})

# ===============================================================
# Authentication Routes (unchanged)
# ===============================================================

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        if not data or "username" not in data or "password" not in data or "email" not in data:
            return jsonify({"error": "Invalid input - username, password and email required"}), 400

        existing_user = User.query.filter((User.username == data['username']) | (User.email == data['email'])).first()
        if existing_user:
            return jsonify({"error": "Username or Email already exists"}), 400

        hashed_pw = generate_password_hash(data['password'])
        new_user = User(username=data['username'], email=data['email'], password_hash=hashed_pw, role='user')
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User created successfully"}), 201

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data or "username" not in data or "password" not in data:
            return jsonify({"error": "Invalid input - username and password required"}), 400

        user = User.query.filter_by(username=data['username']).first()
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({"error": "Invalid username or password"}), 401

        token = jwt.encode(
            {
                "user_id": user.id,
                "username": user.username,
                "role": user.role,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            },
            app.config['SECRET_KEY'],
            algorithm="HS256"
        )

        if isinstance(token, bytes):
            token = token.decode('utf-8')

        return jsonify({
            "token": token,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/profile", methods=["GET", "POST"])
@token_required
def profile(current_user):
    profile = Profile.query.filter_by(user_id=current_user.id).first()

    if request.method == "POST":
        data = request.get_json()
        if not profile:
            profile = Profile(user_id=current_user.id)

        profile.language = data.get("language", profile.language)
        profile.interests = data.get("interests", profile.interests)
        db.session.add(profile)
        db.session.commit()
        return jsonify({"message": "Profile updated!"})

    return jsonify({
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "language": profile.language if profile else None,
        "interests": profile.interests if profile else None
    })

# ===============================================================
# Dataset Management Routes (unchanged)
# ===============================================================

@app.route("/upload", methods=["POST"])
@token_required
def upload_file(current_user):
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
        os.makedirs(user_folder, exist_ok=True)
        filepath = os.path.join(user_folder, filename)
        file.save(filepath)
        return jsonify({"message": f"{filename} uploaded successfully"})

    return jsonify({"error": "File type not allowed"}), 400

@app.route("/datasets", methods=["GET"])
@token_required
def list_datasets(current_user):
    user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    if not os.path.exists(user_folder):
        return jsonify({"datasets": []})
    return jsonify({"datasets": os.listdir(user_folder)})

@app.route("/datasets/<filename>", methods=["DELETE"])
@token_required
def delete_dataset(current_user, filename):
    user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    filepath = os.path.join(user_folder, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({"message": f"{filename} deleted"})
    return jsonify({"error": "File not found"}), 404

@app.route("/datasets/preview/<filename>", methods=["GET"])
@token_required
def preview_dataset(current_user, filename):
    user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    filepath = os.path.join(user_folder, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            return jsonify(df.head().to_dict(orient="records"))
        elif filename.endswith(".json"):
            with open(filepath) as f:
                data = json.load(f)
            return jsonify(data[:5] if isinstance(data, list) else dict(list(data.items())[:5]))
        elif filename.endswith(".txt"):
            with open(filepath) as f:
                lines = [next(f).strip() for _ in range(5)]
            return jsonify(lines)
        else:
            return jsonify({"error": "Preview not supported"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/datasets/preprocess/<filename>", methods=["POST"])
@token_required
def preprocess_dataset(current_user, filename):
    user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    filepath = os.path.join(user_folder, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath).dropna()
            df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
            clean_path = filepath.replace(".", "_cleaned.")
            df.to_csv(clean_path, index=False)
            return jsonify({
                "message": f"Preprocessed CSV saved as {os.path.basename(clean_path)}",
                "preprocessed_preview": df.head().to_dict(orient="records")
            })
        elif filename.endswith(".json"):
            with open(filepath) as f:
                data = json.load(f)
            if isinstance(data, list):
                clean_data = [dict((k, str(v).strip() if isinstance(v, str) else v) for k,v in d.items()) for d in data]
            else:
                clean_data = dict((k, str(v).strip() if isinstance(v, str) else v) for k,v in data.items())
            clean_path = filepath.replace(".", "_cleaned.")
            with open(clean_path, "w") as f:
                json.dump(clean_data, f, indent=2)
            return jsonify({
                "message": f"Preprocessed JSON saved as {os.path.basename(clean_path)}",
                "preprocessed_preview": clean_data if isinstance(clean_data, dict) else clean_data[:5]
            })
        elif filename.endswith(".txt"):
            with open(filepath) as f:
                lines = [l.strip() for l in f if l.strip()]
            clean_path = filepath.replace(".", "_cleaned.")
            with open(clean_path, "w") as f:
                f.write("\n".join(lines))
            return jsonify({
                "message": f"Preprocessed TXT saved as {os.path.basename(clean_path)}",
                "preprocessed_preview": lines[:5]
            })
        else:
            return jsonify({"error": "Preprocessing not supported"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/datasets/download/<filename>", methods=["GET"])
@token_required
def download_dataset(current_user, filename):
    user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    filepath = os.path.join(user_folder, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "file not found"}), 404
    return send_file(filepath, as_attachment=True)

# ===============================================================
# Knowledge Graph Extraction Routes (modified to include Neo4j)
# ===============================================================

def enhanced_csv_triple_extraction(df, filename):
    """Enhanced CSV processing that creates structured triples"""
    triples = []
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    relation_patterns = {
        'location': ['continent', 'region', 'subregion', 'country', 'city', 'location', 'place'],
        'codes': ['alpha_2', 'alpha_3', 'code', 'iso_code', 'country_code', 'numeric_code'],
        'demographics': ['population', 'area', 'density', 'capital', 'language', 'currency']
    }
    relation_mapping = {
        'continent': 'located_in_continent',
        'region': 'located_in_region',
        'subregion': 'located_in_subregion',
        'alpha_2': 'has_alpha_2_code',
        'alpha_3': 'has_alpha_3_code',
        'country_code': 'has_country_code',
        'capital': 'has_capital',
        'language': 'speaks_language',
        'currency': 'uses_currency',
        'population': 'has_population'
    }

    for _, row in df.iterrows():
        if row.isna().all():
            continue

        main_entity = None
        entity_col = None

        if 'name' in df.columns:
            main_entity = str(row['name']).strip()
            entity_col = 'name'
        else:
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    main_entity = str(row[col]).strip()
                    entity_col = col
                    break

        if not main_entity or main_entity.lower() in ['', 'nan', 'null', 'none']:
            continue

        for col, val in row.items():
            if col == entity_col or pd.isna(val) or str(val).strip() in ['', 'nan', 'null', 'none']:
                continue

            val = str(val).strip()
            col_lower = col.lower()

            relation = None

            if col in relation_mapping:
                relation = relation_mapping[col]
            else:
                for pattern_key, pattern_list in relation_patterns.items():
                    if any(pattern in col_lower for pattern in pattern_list):
                        if pattern_key == 'location':
                            relation = f"located_in_{col}" if 'code' not in col_lower else f"has_{col}"
                        elif pattern_key == 'codes':
                            relation = f"has_{col}"
                        elif pattern_key == 'demographics':
                            if 'capital' in col_lower:
                                relation = 'has_capital'
                            elif 'language' in col_lower:
                                relation = 'speaks_language'
                            elif 'currency' in col_lower:
                                relation = 'uses_currency'
                            else:
                                relation = f"has_{col}"
                        break

            if not relation:
                relation = f"has_{col}"

            relation = relation.replace('__', '_').strip('_')

            triples.append((main_entity, relation, val))

    return triples

@app.route("/datasets/extract/<filename>", methods=["POST"])
@token_required
def extract_triples(current_user, filename):
    user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    triples = []
    clean_triples = []

    try:
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text_data = f.read()
            triples.extend(run_rebel(text_data))
            triples.extend(run_dependency_rel(text_data))

        elif filename.endswith(".csv"):
            df = pd.read_csv(filepath).astype(str).fillna("")
            csv_triples = enhanced_csv_triple_extraction(df, filename)
            triples.extend(csv_triples)

        elif filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            text_data = str(data)
            triples.extend(run_rebel(text_data))
            triples.extend(run_dependency_rel(text_data))

        else:
            return jsonify({"error": "Unsupported file type for extraction"}), 400

    except Exception as e:
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500

    seen = set()
    for triple in triples:
        if len(triple) == 3:
            e1, rel, e2 = triple

            e1 = str(e1).strip()
            rel = str(rel).strip()
            e2 = str(e2).strip()

            if not e1 or not rel or not e2:
                continue

            if e1.lower() == e2.lower():
                continue

            e1 = e1.replace('"', '').replace("'", "").strip()
            e2 = e2.replace('"', '').replace("'", "").strip()
            rel = rel.replace('"', '').replace("'", "").strip()

            norm_key = (e1.lower(), rel.lower(), e2.lower())

            if norm_key not in seen:
                clean_triples.append({
                    "entity1": e1,
                    "relation": rel,
                    "entity2": e2,
                    "source": "extraction"
                })
                seen.add(norm_key)

    # Save to both SQLite and Neo4j
    kb_saved = save_triples_to_kb(clean_triples)
    neo4j_saved = save_triples_to_neo4j(clean_triples)

    G = nx.DiGraph()
    for t in clean_triples:
        e1, rel, e2 = t["entity1"], t["relation"], t["entity2"]
        G.add_edge(e1, e2, label=rel, relation=rel, source=t.get("source", "extraction"))

    base_name = os.path.splitext(filename)[0]
    graph_filename = f"{base_name}_graph.gpickle"
    graph_path = os.path.join(user_folder, graph_filename)

    try:
        nx.write_gpickle(G, graph_path)
        graph_saved = True
        print(f"Graph saved: {graph_filename}")
    except Exception as e:
        print(f"Error saving graph: {e}")
        graph_filename = f"{base_name}_graph.json"
        graph_path = os.path.join(user_folder, graph_filename)
        try:
            graph_data = nx.node_link_data(G)
            with open(graph_path, 'w') as f:
                json.dump(graph_data, f)
            graph_saved = True
            print(f"Graph saved as JSON: {graph_filename}")
        except Exception as json_error:
            print(f"Error saving JSON graph: {json_error}")
            graph_saved = False

    search_loaded = False
    search_nodes = 0
    if graph_saved:
        try:
            node_count = search_engine.build_index(G)
            search_loaded = True
            search_nodes = node_count
            print(f"Graph auto-loaded into search: {node_count} nodes")
        except Exception as e:
            print(f"Error auto-loading graph: {e}")
            search_loaded = False

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0.0,
        "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        "graph_file": graph_filename,
        "search_loaded": search_loaded,
        "search_nodes": search_nodes,
        "kb_saved": kb_saved,
        "neo4j_saved": neo4j_saved
    }

    if G.number_of_nodes() > 0:
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
        stats["top_degree_entities"] = [{"node": node, "degree": degree} for node, degree in degrees]
    else:
        stats["top_degree_entities"] = []

    return jsonify({
        "message": f"Extracted {len(clean_triples)} triples. Saved {kb_saved} to SQLite, {neo4j_saved} to Neo4j.",
        "triples": clean_triples,
        "graph_stats": stats
    }), 200

# ===============================================================
# Neo4j Specific Routes
# ===============================================================

@app.route("/neo4j/status", methods=["GET"])
@token_required
def neo4j_status(current_user):
    """Check Neo4j connection status"""
    if not neo4j_conn:
        return jsonify({
            "available": False,
            "message": "Neo4j not configured or driver not available"
        })
    
    try:
        test_result = neo4j_conn.test_connection()
        stats = neo4j_conn.get_graph_stats()
        
        return jsonify({
            "available": True,
            "message": test_result,
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "available": False,
            "message": f"Neo4j connection error: {str(e)}"
        })

@app.route("/neo4j/query", methods=["POST"])
@token_required
def neo4j_query(current_user):
    """Run a Cypher query against Neo4j"""
    if not neo4j_conn:
        return jsonify({"error": "Neo4j not available"}), 400
        
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query required"}), 400
        
    query = data["query"]
    
    try:
        with neo4j_conn.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]
            
            return jsonify({
                "query": query,
                "results": records,
                "count": len(records)
            })
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

@app.route("/neo4j/relations/<entity_name>", methods=["GET"])
@token_required
def neo4j_get_relations(current_user, entity_name):
    """Get all relations for a specific entity"""
    if not neo4j_conn:
        return jsonify({"error": "Neo4j not available"}), 400
        
    try:
        relations = neo4j_conn.query_relations(entity_name)
        
        return jsonify({
            "entity": entity_name,
            "relations": [
                {
                    "source": rel[0],
                    "relation": rel[1],
                    "target": rel[2]
                } for rel in relations
            ],
            "count": len(relations)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get relations: {str(e)}"}), 500

@app.route("/neo4j/triples", methods=["GET"])
@token_required
def neo4j_get_triples(current_user):
    """Get all triples from Neo4j"""
    if not neo4j_conn:
        return jsonify({"error": "Neo4j not available"}), 400
        
    try:
        limit = request.args.get('limit', 100, type=int)
        triples = neo4j_conn.get_all_triples(limit)
        
        return jsonify({
            "triples": [
                {
                    "entity1": triple[0],
                    "relation": triple[1],
                    "entity2": triple[2]
                } for triple in triples
            ],
            "count": len(triples),
            "limit": limit
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get triples: {str(e)}"}), 500

@app.route("/neo4j/add_triple", methods=["POST"])
@token_required
def neo4j_add_triple(current_user):
    """Add a single triple to Neo4j"""
    if not neo4j_conn:
        return jsonify({"error": "Neo4j not available"}), 400
        
    data = request.get_json()
    if not data or "entity1" not in data or "relation" not in data or "entity2" not in data:
        return jsonify({"error": "entity1, relation, and entity2 are required"}), 400
        
    try:
        entity1 = data["entity1"]
        relation = data["relation"]
        entity2 = data["entity2"]
        
        neo4j_conn.add_triple(entity1, relation, entity2)
        
        return jsonify({
            "message": "Triple added successfully",
            "triple": {
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to add triple: {str(e)}"}), 500

@app.route("/neo4j/clear", methods=["POST"])
@token_required
@admin_required
def neo4j_clear(current_user):
    """Clear all data from Neo4j (admin only)"""
    if not neo4j_conn:
        return jsonify({"error": "Neo4j not available"}), 400
        
    try:
        message = neo4j_conn.clear_database()
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": f"Failed to clear database: {str(e)}"}), 500

# ===============================================================
# Semantic Search Routes (unchanged)
# ===============================================================

@app.route("/semantic/status", methods=["GET"])
@token_required
def semantic_status(current_user):
    """Check if semantic search is ready"""
    try:
        user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))
        graph_files = []

        if os.path.exists(user_folder):
            graph_files = [f for f in os.listdir(user_folder) if f.endswith('.gpickle') or f.endswith('_graph.json')]

        return jsonify({
            "graph_available": len(graph_files) > 0,
            "graph_files": graph_files,
            "search_ready": search_engine.graph is not None and search_engine.graph.number_of_nodes() > 0,
            "search_nodes": search_engine.graph.number_of_nodes() if search_engine.graph else 0,
            "search_engine_ready": search_engine.model is not None
        })
    except Exception as e:
        return jsonify({"error": f"Status check failed: {str(e)}"}), 500

@app.route("/semantic/load_graph", methods=["POST"])
@token_required
def load_semantic_graph(current_user):
    """Load the most recent graph for a user into semantic search"""
    try:
        user_folder = os.path.join(UPLOAD_FOLDER, str(current_user.id))

        if not os.path.exists(user_folder):
            return jsonify({"error": "No user folder found"}), 400

        graph_files = [f for f in os.listdir(user_folder) if f.endswith('.gpickle')]

        if not graph_files:
            json_graph_files = [f for f in os.listdir(user_folder) if f.endswith('_graph.json')]
            if not json_graph_files:
                return jsonify({"error": "No graph files found"}), 400
            graph_files = json_graph_files

        graph_files_with_paths = [(f, os.path.join(user_folder, f)) for f in graph_files]
        graph_files_with_mtime = [(f, path, os.path.getmtime(path)) for f, path in graph_files_with_paths if os.path.exists(path)]

        if not graph_files_with_mtime:
            return jsonify({"error": "No valid graph files found"}), 400

        graph_files_with_mtime.sort(key=lambda x: x[2], reverse=True)
        latest_file, latest_path, mtime = graph_files_with_mtime[0]

        print(f"Loading graph: {latest_file}")

        if latest_file.endswith('.gpickle'):
            G = nx.read_gpickle(latest_path)
        else:
            with open(latest_path, 'r') as f:
                graph_data = json.load(f)
            G = nx.node_link_graph(graph_data)

        node_count = search_engine.build_index(G)

        return jsonify({
            "message": f"Graph '{latest_file}' loaded successfully with {node_count} nodes",
            "nodes_loaded": node_count,
            "graph_file": latest_file
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load graph: {str(e)}"}), 500

@app.route("/semantic/search", methods=["POST"])
@token_required
def semantic_search(current_user):
    """Perform semantic search on knowledge graph"""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query required"}), 400

    query = data["query"].strip()
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if search_engine.graph is None:
        return jsonify({"error": "No graph loaded. Please load a graph first."}), 400

    if search_engine.model is None:
        return jsonify({"error": "Semantic search model not loaded"}), 400

    try:
        results = search_engine.search_nodes(query, top_k)

        return jsonify({
            "query": query,
            "results": results,
            "total_found": len(results)
        })
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route("/semantic/subgraph", methods=["POST"])
@token_required
def semantic_subgraph(current_user):
    """Generate subgraph from semantic search results"""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query required"}), 400

    query = data["query"].strip()
    top_k = data.get("top_k", 3)
    radius = data.get("radius", 1)

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if search_engine.graph is None:
        return jsonify({"error": "No graph loaded. Please load a graph first."}), 400

    try:
        subgraph, top_nodes = search_engine.search_to_subgraph(query, top_k, radius)

        response_data = {
            "query": query,
            "top_nodes": top_nodes
        }

        if subgraph is not None:
            subgraph_data = nx.node_link_data(subgraph)
            response_data.update({
                "subgraph": subgraph_data,
                "node_count": subgraph.number_of_nodes(),
                "edge_count": subgraph.number_of_edges(),
                "subgraph_generated": True
            })
        else:
            response_data.update({
                "subgraph": None,
                "node_count": 0,
                "edge_count": 0,
                "subgraph_generated": False
            })

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Subgraph generation failed: {str(e)}"}), 500

# ===============================================================
# MILESTONE 4: Admin Dashboard & Feedback System (SECURED) - Updated with Neo4j
# ===============================================================

@app.route("/admin/register", methods=["POST"])
def register_admin():
    """Create the first admin user (one-time setup)"""
    try:
        data = request.get_json()
        if not data or "username" not in data or "password" not in data or "email" not in data:
            return jsonify({"error": "Username, password and email required"}), 400

        existing_admin = User.query.filter_by(role='admin').first()
        if existing_admin:
            return jsonify({"error": "Admin user already exists"}), 400
        
        existing_user = User.query.filter((User.username == data['username']) | (User.email == data['email'])).first()
        if existing_user:
            return jsonify({"error": "Username or Email already exists"}), 400

        hashed_pw = generate_password_hash(data['password'])
        admin_user = User(
            username=data['username'], 
            email=data['email'], 
            password_hash=hashed_pw,
            role='admin'
        )
        db.session.add(admin_user)
        db.session.commit()

        return jsonify({"message": "Admin user created successfully"}), 201

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/admin")
@token_required
@admin_required
def admin_home(current_user):
    """Admin dashboard home page - ONLY for admin users"""
    return render_template('admin_dashboard.html')

@app.route("/admin/stats")
@token_required
@admin_required
def admin_stats(current_user):
    """Get system statistics for admin dashboard - ONLY for admin users"""
    stats = {}

    try:
        conn = sqlite3.connect('knowledge_base.db')
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM triples")
        stats['total_triples'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT entity1 AS entity FROM triples 
                UNION 
                SELECT entity2 AS entity FROM triples
            )
        """)
        stats['total_entities'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT relation) FROM triples")
        stats['total_relations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM triples WHERE created_at >= datetime('now', '-7 days')")
        stats['recent_triples'] = cursor.fetchone()[0]

        conn.close()
    except Exception as e:
        print(f"KB stats error: {e}")
        stats.update({
            'total_triples': 0,
            'total_entities': 0,
            'total_relations': 0,
            'recent_triples': 0
        })

    # Add Neo4j stats
    try:
        if neo4j_conn:
            neo4j_stats = neo4j_conn.get_graph_stats()
            stats['neo4j_nodes'] = neo4j_stats['nodes']
            stats['neo4j_relationships'] = neo4j_stats['relationships']
            stats['neo4j_entities'] = neo4j_stats['entities']
        else:
            stats.update({
                'neo4j_nodes': 0,
                'neo4j_relationships': 0,
                'neo4j_entities': 0
            })
    except Exception as e:
        print(f"Neo4j stats error: {e}")
        stats.update({
            'neo4j_nodes': 0,
            'neo4j_relationships': 0,
            'neo4j_entities': 0
        })

    try:
        feedback_file = 'feedback.csv'
        if os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)

            if not feedback_df.empty:
                if 'rating' in feedback_df.columns:
                    feedback_df['rating'] = pd.to_numeric(feedback_df['rating'], errors='coerce')
                    avg_rating = round(feedback_df['rating'].mean(skipna=True), 2)
                    stats['avg_rating'] = 0 if pd.isna(avg_rating) else avg_rating
                else:
                    stats['avg_rating'] = 0.0

                stats['total_feedback'] = len(feedback_df)

                if 'timestamp' in feedback_df.columns:
                    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'], errors='coerce')
                    week_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
                    recent_feedback = len(feedback_df[feedback_df['timestamp'] >= week_ago])
                    stats['recent_feedback'] = int(recent_feedback)
                else:
                    stats['recent_feedback'] = 0
            else:
                stats.update({'total_feedback': 0, 'avg_rating': 0.0, 'recent_feedback': 0})
        else:
            stats.update({'total_feedback': 0, 'avg_rating': 0.0, 'recent_feedback': 0})

    except Exception as e:
        print(f"Feedback stats error: {e}")
        stats.update({'total_feedback': 0, 'avg_rating': 0.0, 'recent_feedback': 0})

    try:
        stats['total_users'] = User.query.count()
        stats['admin_users'] = User.query.filter_by(role='admin').count()
        stats['regular_users'] = User.query.filter_by(role='user').count()
        stats['active_users'] = stats['total_users']  # placeholder
    except Exception as e:
        print(f"User stats error: {e}")
        stats.update({
            'total_users': 0,
            'admin_users': 0,
            'regular_users': 0,
            'active_users': 0
        })

    print("📊 Admin Stats:", stats)
    return jsonify(stats)

@app.route("/admin/neo4j")
@token_required
@admin_required
def admin_neo4j(current_user):
    """Admin Neo4j management page"""
    neo4j_stats = {}
    if neo4j_conn:
        try:
            neo4j_stats = neo4j_conn.get_graph_stats()
        except:
            neo4j_stats = {"error": "Failed to get Neo4j stats"}
    
    return render_template('admin_neo4j.html', neo4j_stats=neo4j_stats, neo4j_available=bool(neo4j_conn))

@app.route("/admin/kb")
@token_required
@admin_required
def view_kb(current_user):
    """View all knowledge base entries - ONLY for admin users"""
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triples ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return render_template('view_kb.html', data=data)

@app.route("/admin/kb/add", methods=["GET", "POST"])
@token_required
@admin_required
def add_triple(current_user):
    """Add new knowledge triple - ONLY for admin users"""
    if request.method == 'POST':
        entity1 = request.form['entity1']
        relation = request.form['relation']
        entity2 = request.form['entity2']
        
        conn = sqlite3.connect('knowledge_base.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO triples (entity1, relation, entity2) VALUES (?, ?, ?)",
            (entity1, relation, entity2)
        )
        conn.commit()
        conn.close()
        
        # Also add to Neo4j if available
        if neo4j_conn:
            try:
                neo4j_conn.add_triple(entity1, relation, entity2)
            except Exception as e:
                print(f"Failed to add to Neo4j: {e}")
        
        return redirect('/admin/kb')
    
    return render_template('add_triple.html')

@app.route("/admin/kb/edit/<int:id>", methods=["GET", "POST"])
@token_required
@admin_required
def edit_triple(current_user, id):
    """Edit knowledge triple - ONLY for admin users"""
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        entity1 = request.form['entity1']
        relation = request.form['relation']
        entity2 = request.form['entity2']
        
        cursor.execute(
            "UPDATE triples SET entity1=?, relation=?, entity2=? WHERE id=?",
            (entity1, relation, entity2, id)
        )
        conn.commit()
        conn.close()
        return redirect('/admin/kb')
    
    cursor.execute("SELECT * FROM triples WHERE id=?", (id,))
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        return "Record not found", 404
        
    return render_template('edit_triple.html', record=record)

@app.route("/admin/kb/delete/<int:id>")
@token_required
@admin_required
def delete_triple(current_user, id):
    """Delete knowledge triple - ONLY for admin users"""
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM triples WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect('/admin/kb')

@app.route("/admin/feedback")
@token_required
@admin_required
def view_feedback(current_user):
    """View user feedback - ONLY for admin users"""
    try:
        if os.path.exists('feedback.csv'):
            df = pd.read_csv('feedback.csv')

            if df.empty:
                feedback_data = []
            else:
                feedback_data = df.to_dict('records')
        else:
            feedback_data = []
    except Exception as e:
        print(f"Feedback load error: {e}")
        feedback_data = []
        
    return render_template('view_feedback.html', feedback=feedback_data)

# ===============================================================
# Feedback System Routes (unchanged)
# ===============================================================

@app.route("/feedback", methods=["POST"])
@token_required
def submit_feedback(current_user):
    """Store user feedback (rating + comments) into feedback.csv"""
    import traceback
    try:
        data = request.get_json(force=True, silent=True)
        print(" Received feedback payload:", data)

        if not isinstance(data, dict):
            print("❌ Invalid JSON or empty request body")
            return jsonify({"error": "Invalid feedback data"}), 400

        username = getattr(current_user, "username", data.get("username", "Anonymous"))
        rating = int(data.get("rating", 0))
        comment = data.get("comment") or data.get("comments", "")
        query = data.get("query", "")
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


        feedback_file = os.path.join(os.getcwd(), "feedback.csv")
        print(f"📂 Writing feedback file to: {feedback_file}")

        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

        file_exists = os.path.exists(feedback_file)
        with open(feedback_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["username", "query", "rating", "comments", "timestamp"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "username": username,
                "query": query,
                "rating": rating,
                "comments": comment,
                "timestamp": timestamp
            })

        print("✅ Feedback saved successfully.")
        return jsonify({"message": "Feedback submitted successfully"}), 201

    except Exception as e:
        print("❌ Feedback submission error:", e)
        traceback.print_exc()
        return jsonify({"error": f"Error saving feedback: {str(e)}"}), 500

@app.route("/feedback/stats", methods=["GET"])
@token_required
def feedback_stats(current_user):
    """Get feedback statistics (total count + average rating)"""
    try:
        feedback_file = "feedback.csv"
        if not os.path.exists(feedback_file):
            return jsonify({
                "total_feedback": 0,
                "average_rating": 0
            })

        df = pd.read_csv(feedback_file)

        if df.empty:
            return jsonify({
                "total_feedback": 0,
                "average_rating": 0
            })

        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            avg_rating = round(df["rating"].mean(skipna=True), 2)
        else:
            avg_rating = 0

        stats = {
            "total_feedback": len(df),
            "average_rating": 0 if pd.isna(avg_rating) else avg_rating
        }

        return jsonify(stats), 200

    except Exception as e:
        print(f"❌ Feedback stats error: {e}")
        return jsonify({"error": f"Failed to get stats: {str(e)}"}), 500

# ===============================================================
# Enhanced CRUD API Endpoints for Streamlit (unchanged)
# ===============================================================

@app.route("/api/kb/triples", methods=["GET"])
@token_required
@admin_required
def api_get_triples(current_user):
    """Get all knowledge base triples as JSON"""
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triples ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    
    triples = []
    for row in data:
        triples.append({
            "id": row[0],
            "entity1": row[1],
            "relation": row[2],
            "entity2": row[3],
            "created_at": row[4]
        })
    
    return jsonify({"triples": triples})

@app.route("/api/kb/triples", methods=["POST"])
@token_required
@admin_required
def api_add_triple(current_user):
    """Add a new triple via API"""
    try:
        data = request.get_json()
        if not data or "entity1" not in data or "relation" not in data or "entity2" not in data:
            return jsonify({"error": "entity1, relation, and entity2 are required"}), 400

        entity1 = data["entity1"].strip()
        relation = data["relation"].strip()
        entity2 = data["entity2"].strip()

        if not entity1 or not relation or not entity2:
            return jsonify({"error": "All fields must be non-empty"}), 400

        conn = sqlite3.connect('knowledge_base.db')
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id FROM triples WHERE entity1=? AND relation=? AND entity2=?",
            (entity1, relation, entity2)
        )
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return jsonify({"error": "Triple already exists"}), 400

        # Insert new triple
        cursor.execute(
            "INSERT INTO triples (entity1, relation, entity2) VALUES (?, ?, ?)",
            (entity1, relation, entity2)
        )
        conn.commit()
        triple_id = cursor.lastrowid
        conn.close()

        # Also add to Neo4j
        if neo4j_conn:
            try:
                neo4j_conn.add_triple(entity1, relation, entity2)
            except Exception as e:
                print(f"Failed to add to Neo4j: {e}")

        return jsonify({
            "message": "Triple added successfully",
            "id": triple_id,
            "triple": {
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2
            }
        }), 201

    except Exception as e:
        return jsonify({"error": f"Failed to add triple: {str(e)}"}), 500

@app.route("/api/kb/triples/<int:triple_id>", methods=["DELETE"])
@token_required
@admin_required
def api_delete_triple(current_user, triple_id):
    """Delete a triple via API"""
    try:
        conn = sqlite3.connect('knowledge_base.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM triples WHERE id=?", (triple_id,))
        existing = cursor.fetchone()
        
        if not existing:
            conn.close()
            return jsonify({"error": "Triple not found"}), 404

        cursor.execute("DELETE FROM triples WHERE id=?", (triple_id,))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Triple {triple_id} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to delete triple: {str(e)}"}), 500

@app.route("/api/kb/triples/<int:triple_id>", methods=["PUT"])
@token_required
@admin_required
def api_update_triple(current_user, triple_id):
    """Update a triple via API"""
    try:
        data = request.get_json()
        if not data or "entity1" not in data or "relation" not in data or "entity2" not in data:
            return jsonify({"error": "entity1, relation, and entity2 are required"}), 400

        entity1 = data["entity1"].strip()
        relation = data["relation"].strip()
        entity2 = data["entity2"].strip()

        if not entity1 or not relation or not entity2:
            return jsonify({"error": "All fields must be non-empty"}), 400

        conn = sqlite3.connect('knowledge_base.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM triples WHERE id=?", (triple_id,))
        existing = cursor.fetchone()
        
        if not existing:
            conn.close()
            return jsonify({"error": "Triple not found"}), 404

        cursor.execute(
            "UPDATE triples SET entity1=?, relation=?, entity2=? WHERE id=?",
            (entity1, relation, entity2, triple_id)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "message": f"Triple {triple_id} updated successfully",
            "triple": {
                "id": triple_id,
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to update triple: {str(e)}"}), 500

@app.route("/api/kb/search", methods=["GET"])
@token_required
@admin_required
def api_search_triples(current_user):
    """Search triples via API"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({"error": "Search query required"}), 400

        conn = sqlite3.connect('knowledge_base.db')
        cursor = conn.cursor()
        
        search_pattern = f"%{query}%"
        cursor.execute("""
            SELECT * FROM triples 
            WHERE entity1 LIKE ? OR relation LIKE ? OR entity2 LIKE ?
            ORDER BY id DESC
        """, (search_pattern, search_pattern, search_pattern))
        
        data = cursor.fetchall()
        conn.close()
        
        triples = []
        for row in data:
            triples.append({
                "id": row[0],
                "entity1": row[1],
                "relation": row[2],
                "entity2": row[3],
                "created_at": row[4]
            })
        
        return jsonify({
            "query": query,
            "results": triples,
            "count": len(triples)
        })

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

# ===============================================================
# Application Startup
# ===============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=False)