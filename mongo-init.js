// MongoDB initialization script
db = db.getSiblingDB('retirement_chatbot');

// Create collections with indexes
db.createCollection('sessions');
db.createCollection('messages');

// Create indexes for better performance
db.sessions.createIndex({ "user_id": 1 });
db.sessions.createIndex({ "session_id": 1 }, { unique: true });
db.sessions.createIndex({ "updated_at": -1 });

db.messages.createIndex({ "session_id": 1 });
db.messages.createIndex({ "timestamp": 1 });
db.messages.createIndex({ "id": 1 }, { unique: true });

print('Database initialized successfully');