import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // Navigation: 'menu' | 'ticket' | 'status' | 'view' | 'comment' | 'chatbot' | 'result'
  const [currentStep, setCurrentStep] = useState('menu');
  const [loading, setLoading] = useState(false);
  const [agentResponse, setAgentResponse] = useState('');

  // 🛡️ MASKED LAB DEFAULTS [For the time being it is SCRUM]
  const DEFAULT_PROJECT_KEY = "SCRUM";
  const DEFAULT_USER_FALLBACK = "me"; 

  // Form Fields States (Project Key is completely hidden from UI forms)
  const [taskName, setTaskName] = useState('');
  const [taskDescription, setTaskDescription] = useState('');
  const [assignee, setAssignee] = useState('');
  const [watcher, setWatcher] = useState('');
  const [ticketKey, setTicketKey] = useState('');
  const [targetStatus, setTargetStatus] = useState('In Progress');
  const [commentText, setCommentText] = useState('');

  // Conversational Chatbot States
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { role: 'agent', text: "🤖 Hello! I am your direct lab assistant. Ask me to make tasks, change columns, or check details on your SCRUM board using natural language commands." }
  ]);

  // Form submit handles that inject defaults behind the scenes before hitting Python
  const handleTicketCreation = async (e) => {
    e.preventDefault();
    if (!taskName.trim()) return;

    // Apply Defaults: If empty, use 'me' string so Python tool falls back to your account
    const finalAssignee = assignee.trim() ? assignee : DEFAULT_USER_FALLBACK;
    const finalWatcher = watcher.trim() ? watcher : DEFAULT_USER_FALLBACK;

    let complexPrompt = `In project workspace "${DEFAULT_PROJECT_KEY}", create a new task. Summary: "${taskName}". Description: "${taskDescription}".`;
    complexPrompt += ` Assign this new ticket directly to "${finalAssignee}".`;
    complexPrompt += ` Add "${finalWatcher}" as an active watcher monitor to this ticket.`;

    await submitToAgent(complexPrompt, 'form');
  };

  const handleStatusTransition = async (e) => {
    e.preventDefault();
    if (!ticketKey.trim()) return;
    
    // Auto-appends hidden project prefix if the user only types a single number
    const formattedKey = ticketKey.toUpperCase().includes(DEFAULT_PROJECT_KEY) 
      ? ticketKey.toUpperCase() 
      : `${DEFAULT_PROJECT_KEY}-${ticketKey.trim()}`;

    const complexPrompt = `Find existing ticket "${formattedKey}" and move its workflow status to "${targetStatus}".`;
    await submitToAgent(complexPrompt, 'form');
  };
  const handleViewTicket = async (e) => {
    e.preventDefault();
    if (!ticketKey.trim()) return;
    const formattedKey = ticketKey.toUpperCase().includes(DEFAULT_PROJECT_KEY) ? ticketKey.toUpperCase() : `${DEFAULT_PROJECT_KEY}-${ticketKey.trim()}`;
    const complexPrompt = `Retrieve and show me the status, summary, and comments of ticket "${formattedKey}".`;
    await submitToAgent(complexPrompt, 'form');
  };

  const handleCommentSubmission = async (e) => {
    e.preventDefault();
    if (!ticketKey.trim() || !commentText.trim()) return;
    const formattedKey = ticketKey.toUpperCase().includes(DEFAULT_PROJECT_KEY) ? ticketKey.toUpperCase() : `${DEFAULT_PROJECT_KEY}-${ticketKey.trim()}`;
    const complexPrompt = `Find existing ticket "${formattedKey}" and add a comment saying "${commentText}".`;
    await submitToAgent(complexPrompt, 'form');
  };

  // Conversational Chatbot Submit Action Handler
  const handleChatbotSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || loading) return;

    const userQuery = chatInput;
    setChatHistory(prev => [...prev, { role: 'user', text: userQuery }]);
    setChatInput('');
    
    await submitToAgent(userQuery, 'chat');
  };

  const submitToAgent = async (promptText, contextMode) => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/chat', { message: promptText });
      const backendReply = res.data.response;

      if (contextMode === 'chat') {
        setChatHistory(prev => [...prev, { role: 'agent', text: backendReply }]);
      } else {
        setAgentResponse(backendReply);
        setCurrentStep('result');
      }
    } catch (error) {
      console.error(error);
      const fallbackError = '❌ Communication failure connecting to your main.py python API port.';
      if (contextMode === 'chat') {
        setChatHistory(prev => [...prev, { role: 'agent', text: fallbackError }]);
      } else {
        setAgentResponse(fallbackError);
        setCurrentStep('result');
      }
    } finally {
      setLoading(false);
    }
  };

    const resetWizard = () => {
    setTaskName('');
    setTaskDescription('');
    setAssignee('');
    setWatcher('');
    setTicketKey('');
    setCommentText('');
    setAgentResponse('');
    setCurrentStep('menu');
  }; // 👈 This closing brace MUST be here to properly close resetWizard

  // 🧹 NEW: Flushes the conversational chatbot thread memory (Declared separately)
  const clearChatHistory = () => {
    setChatHistory([
      { role: 'agent', text: "🤖 Chat memory cleared. Hello! Ask me to create tasks, change columns, or check details on your SCRUM board using natural language commands." }
    ]);
    setChatInput('');
  };



  return (
    <div className="lab-workspace">
      <aside className="capabilities-sidebar">
        <h3>🧠 AI Agent Panel</h3>
        <p>Your workspace is hard-configured with direct background automation variables.</p>
        <ul>
          <li>🔒 <strong>Active Key:</strong> Locked to SCRUM</li>
          <li>🛡️ <strong>Auto-Fallback:</strong> Enabled to Active Profile</li>
          <li>💬 <strong>Omni-Chatbot:</strong> Live Stream Mode Active</li>
        </ul>
        <div className="system-pill">Active LLM: Llama 3.1</div>
      </aside>

      <main className="wizard-core">
        <header className="wizard-header">
          <h2>🛠️ Jira AI Assistant Console</h2>
          <p>Switch between individual form fields or conversational chatbot mode below.</p>
        </header>

        {currentStep === 'menu' && (
          <div className="menu-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <button className="menu-card" onClick={() => setCurrentStep('ticket')}>
              <span className="card-icon">🎫</span>
              <h4>Build & Assign Ticket</h4>
            </button>
            <button className="menu-card" onClick={() => setCurrentStep('status')}>
              <span className="card-icon">🔄</span>
              <h4>Shift Ticket Status</h4>
            </button>
            <button className="menu-card" onClick={() => setCurrentStep('view')}>
              <span className="card-icon">🔍</span>
              <h4>View Ticket Details</h4>
            </button>
            <button className="menu-card" onClick={() => setCurrentStep('comment')}>
              <span className="card-icon">💬</span>
              <h4>Add Ticket Comment</h4>
            </button>
            <button className="menu-card" onClick={() => setCurrentStep('chatbot')} style={{ gridColumn: 'span 2', background: '#e6f4ff', borderColor: '#91d5ff' }}>
              <span className="card-icon">🤖</span>
              <h4>Open Conversational Chatbot Engine</h4>
              <p>Type continuous raw natural language sentences without form layouts.</p>
            </button>
          </div>
        )}

        {currentStep === 'ticket' && (
          <form onSubmit={handleTicketCreation} className="wizard-form">
            <h3>🎫 Step-by-Step Ticket Configuration (Board: SCRUM)</h3>
            <div className="input-group">
              <label>Task Name (Summary)</label>
              <input type="text" placeholder="e.g., Fix broken login layout" value={taskName} onChange={(e) => setTaskName(e.target.value)} required />
            </div>
            <div className="input-group">
              <label>Detailed Description</label>
              <textarea placeholder="Provide background constraints..." value={taskDescription} onChange={(e) => setTaskDescription(e.target.value)} />
            </div>
            <div className="input-row">
              <div className="input-group">
                <label>Assignee Name (Leave empty to assign to yourself)</label>
                <input type="text" placeholder="Blank maps to your profile" value={assignee} onChange={(e) => setAssignee(e.target.value)} />
              </div>
              <div className="input-group">
                <label>Watcher Name (Leave empty to watch yourself)</label>
                <input type="text" placeholder="Blank maps to your profile" value={watcher} onChange={(e) => setWatcher(e.target.value)} />
              </div>
            </div>
            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => setCurrentStep('menu')}>Back</button>
              <button type="submit" className="btn-primary" disabled={loading}>Dispatch AI Agent</button>
            </div>
          </form>
        )}

        {currentStep === 'status' && (
          <form onSubmit={handleStatusTransition} className="wizard-form">
            <h3>🔄 Step-by-Step Status Modifier</h3>
            <div className="input-group">
              <label>Target Ticket (Type key name or just the number number like '1')</label>
              <input type="text" placeholder="e.g., 1" value={ticketKey} onChange={(e) => setTicketKey(e.target.value)} required />
            </div>
            <div className="input-group">
              <label>Target Status Position</label>
              <select value={targetStatus} onChange={(e) => setTargetStatus(e.target.value)}>
                <option value="To Do">To Do</option>
                <option value="In Progress">In Progress</option>
                <option value="Done">Done</option>
              </select>
            </div>
            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => setCurrentStep('menu')}>Back</button>
              <button type="submit" className="btn-primary" disabled={loading}>Update Status</button>
            </div>
          </form>
        )}

        {currentStep === 'view' && (
          <form onSubmit={handleViewTicket} className="wizard-form">
            <h3>🔍 Step-by-Step Ticket Inspector</h3>
            <div className="input-group">
              <label>Target Ticket ID Key/Number</label>
              <input type="text" placeholder="e.g., 1" value={ticketKey} onChange={(e) => setTicketKey(e.target.value)} required />
            </div>
            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => setCurrentStep('menu')}>Back</button>
              <button type="submit" className="btn-primary" disabled={loading}>Inspect Ticket</button>
            </div>
          </form>
        )}

        {currentStep === 'comment' && (
          <form onSubmit={handleCommentSubmission} className="wizard-form">
            <h3>💬 Step-by-Step Comment Logger</h3>
            <div className="input-group">
              <label>Target Ticket ID Key/Number</label>
              <input type="text" placeholder="e.g., 1" value={ticketKey} onChange={(e) => setTicketKey(e.target.value)} required />
            </div>
            <div className="input-group">
              <label>Your Comment Message</label>
              <textarea placeholder="Type troubleshooting updates here..." value={commentText} onChange={(e) => setCommentText(e.target.value)} required />
            </div>
            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => setCurrentStep('menu')}>Back</button>
              <button type="submit" className="btn-primary" disabled={loading}>Add Comment</button>
            </div>
          </form>
        )}

                {currentStep === 'chatbot' && (
          <div className="wizard-form" style={{ height: '65vh', display: 'flex', flexDirection: 'column' }}>
            <h3>🤖 Live AI Agent Omnidirectional Chatbot Console</h3>
            <div className="chat-history-container" style={{ flex: 1, overflowY: 'auto', border: '1px solid #ddd', padding: '15px', borderRadius: '4px', marginBottom: '15px', background: '#fafbfc', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {chatHistory.map((msg, i) => (
                <div 
                  key={i} 
                  style={{ 
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', 
                    background: msg.role === 'user' ? '#deebff' : '#ffffff', 
                    color: msg.role === 'user' ? '#0747a6' : '#333', 
                    padding: '12px 16px', 
                    borderRadius: '8px', 
                    maxWidth: '80%', 
                    border: msg.role === 'user' ? 'none' : '1px solid #ddd',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.02)'
                  }}
                >
                  <p style={{ fontSize: '14px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>
                    {msg.text}
                  </p>
                </div>
              ))}
              {loading && <div style={{ color: '#888', fontStyle: 'italic', padding: '5px' }}>Agent is reasoning and executing tools...</div>}
            </div>
                        {/* Look for your form onSubmit block and match this complete structure */}
            <form onSubmit={handleChatbotSubmit} style={{ display: 'flex', gap: '10px' }}>
              <input 
                type="text" 
                placeholder="e.g., Change status of ticket 1 to Done..." 
                value={chatInput} 
                onChange={(e) => setChatInput(e.target.value)} 
                disabled={loading} 
                style={{ flex: 1 }} 
              />
              <button type="submit" className="btn-primary" disabled={loading}>Send</button>
              
              {/* 🧹 THE SOLUTION: Links your button to clearChatHistory and removes your ESLint warning */}
              <button 
                type="button" 
                className="btn-secondary" 
                style={{ background: '#fff1f0', color: '#cf1322', borderColor: '#ffa39e' }}
                onClick={clearChatHistory} 
                disabled={loading}
              >
                Clear Chat
              </button>
              
              <button type="button" className="btn-secondary" onClick={() => setCurrentStep('menu')}>Exit Chat</button>
            </form>

          </div>
        )}


        {loading && currentStep !== 'chatbot' && (
          <div className="execution-overlay">
            <div className="spinner"></div>
            <h4>AI Agent Active...</h4>
            <p>Processing metrics and modifying your live Jira sandbox.</p>
          </div>
        )}

        {currentStep === 'result' && (
          <div className="result-panel">
            <h3>🏁 Agent Execution Feedback</h3>
            <div className="feedback-content">{agentResponse}</div>
            <button className="btn-primary" onClick={resetWizard}>Return to Menu</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;