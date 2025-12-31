// Add this function to your chatbot.html JavaScript
function renderMarkdown(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **bold**
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // *italic*
        .replace(/\n/g, '<br>');                           // line breaks
}

// Update your message display function to use innerHTML instead of textContent
function displayMessage(message, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'user-message' : 'bot-message';
    
    if (isUser) {
        messageDiv.textContent = message;
    } else {
        // Use innerHTML for bot messages to render markdown
        messageDiv.innerHTML = renderMarkdown(message);
    }
    
    document.getElementById('chat-messages').appendChild(messageDiv);
}