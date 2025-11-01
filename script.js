// Voice Input using Web Speech API (Free)
class VoiceInputHandler {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.initializeSpeechRecognition();
    }

    initializeSpeechRecognition() {
        // Check if browser supports Web Speech API
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            console.error('Speech recognition not supported in this browser');
            this.showError('Voice input is not supported in your browser. Please use Chrome, Edge, or Safari.');
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        // Event handlers
        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateVoiceButton(true);
            this.showStatus('Listening... Speak now');
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById('userInput').value = transcript;
            this.handleVoiceInput(transcript);
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            if (event.error === 'no-speech') {
                this.showStatus('No speech detected. Please try again.');
            } else if (event.error === 'audio-capture') {
                this.showStatus('No microphone found. Please check your microphone.');
            } else if (event.error === 'not-allowed') {
                this.showStatus('Microphone permission denied. Please allow microphone access.');
            } else {
                this.showStatus('Error: ' + event.error);
            }
            this.updateVoiceButton(false);
            this.isListening = false;
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.updateVoiceButton(false);
        };
    }

    startListening() {
        if (!this.recognition) {
            this.showError('Speech recognition not available');
            return;
        }

        if (this.isListening) {
            this.stopListening();
            return;
        }

        try {
            this.recognition.start();
        } catch (error) {
            console.error('Error starting recognition:', error);
            this.showStatus('Error starting voice input. Please try again.');
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
            this.isListening = false;
            this.updateVoiceButton(false);
        }
    }

    updateVoiceButton(isListening) {
        const voiceBtn = document.getElementById('voiceBtn');
        if (isListening) {
            voiceBtn.classList.add('listening');
        } else {
            voiceBtn.classList.remove('listening');
        }
    }

    showStatus(message) {
        const statusEl = document.getElementById('voiceStatus');
        statusEl.textContent = message;
        statusEl.style.display = 'block';
        
        if (message && !message.includes('Error') && !message.includes('permission')) {
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 3000);
        }
    }

    showError(message) {
        this.showStatus(message);
        alert(message); // Fallback alert for important errors
    }

    handleVoiceInput(transcript) {
        // Auto-send the voice input as a message
        const userInput = document.getElementById('userInput');
        userInput.value = transcript;
        this.showStatus('Voice input received!');
        
        // Automatically send the message
        setTimeout(() => {
            sendMessage();
        }, 500);
    }
}

// Chatbot functionality
let voiceHandler;

window.addEventListener('DOMContentLoaded', () => {
    voiceHandler = new VoiceInputHandler();
    
    // Voice button click handler
    const voiceBtn = document.getElementById('voiceBtn');
    voiceBtn.addEventListener('click', () => {
        voiceHandler.startListening();
    });

    // Send button click handler
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.addEventListener('click', sendMessage);

    // Enter key handler (Shift+Enter for new line)
    const userInput = document.getElementById('userInput');
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });
});

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (!message) {
        return;
    }

    // Stop voice recognition if active
    if (voiceHandler && voiceHandler.isListening) {
        voiceHandler.stopListening();
    }

    // Add user message to chat
    addMessageToChat(message, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show typing indicator
    showTypingIndicator();

    try {
        // Call your chatbot API here
        const response = await callChatbotAPI(message);
        
        // Remove typing indicator
        removeTypingIndicator();
	speak(response);
	// Speak text using browser's speech synthesis

	
        
        // Add bot response to chat
        addMessageToChat(response, 'bot');
    } catch (error) {
        console.error('Error calling chatbot:', error);
        removeTypingIndicator();
        addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

function addMessageToChat(message, sender) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showTypingIndicator() {
    const chatContainer = document.getElementById('chatContainer');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = '<div class="message-content"><span></span><span></span><span></span></div>';
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    

    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}
function speak(text) {
    if (!window.speechSynthesis) {
        console.warn("Speech synthesis not supported in this browser.");
        return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-IN";   // Indian English voice
    utterance.rate = 1;         // Normal speed
    utterance.pitch = 1;        // Normal tone

    const voices = window.speechSynthesis.getVoices();
    if (voices.length > 0) {
        utterance.voice = voices.find(v => v.lang === "en-IN") || voices[0];
    }

    window.speechSynthesis.speak(utterance);
}

async function callChatbotAPI(message) {
    // TODO: Replace this with your actual LLaMA chatbot API endpoint
    // Example API call structure:
    
    try {
        // Option 1: If you have a local API endpoint
        const response = await fetch('http://127.0.0.1:5000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();
        return data.response || data.message || 'I received your message.';
        
    } catch (error) {
        // If API is not available, use a placeholder response
        console.warn('API not available, using placeholder response:', error);
        
        // Placeholder response (remove this when your API is ready)
        return `I heard you say: "${message}". This is a placeholder response. Please connect your LLaMA chatbot API in the callChatbotAPI function.`;
    }
}

