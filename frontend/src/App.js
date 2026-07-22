import React, {useState, useRef, useEffect} from 'react'
import './App.css';

function App() {
  // Track what user is currently typing
  const [inputText, setInputText] = useState("")
  // Full chat history from user and bot
  const [messages, setMessages] = useState([])
  // Controls "Detectando dialecto..." state while waiting for API
  const [isLoading, setIsLoading] = useState(false)
  // Reference to invisible element at bottom of chat for auto-scrolling
  const bottomRef = useRef(null)
  useEffect(() => {
  bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleSubmit(){
    // Reject short inputs, model needs at least 4 words to find a pattern
    if (inputText.split(" ").length < 4) {
      setMessages(prev => [...prev, {type: "bot", text: "Please type more than 4 words so we can find a pattern :)"}])
      return
    }

    // Shows user message so the chat feels responsive
    setMessages(prev => [...prev, {type: "user", text: inputText}])
    setIsLoading(true)
    setInputText("")

    // Send sentence to FastAPI backend and wait for prediction
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sentence: inputText})
    })
    const data = await response.json()

    // Format bot reponse with dialect and confidence scores
    const botText = `hmmm based on our calculations, we think it is.... ${data.predicted_dialect}! 
    MX: ${(data.mx_probability * 100).toFixed(1)}% | ES: ${(data.es_probability * 100).toFixed(1)}%
    Here's why we predict this: ${data.explanation}`

    setMessages(prev => [...prev, {type: "bot", text: botText}])
    setIsLoading(false)

  }
  
  return( 
    <div className="app">
      <h1>DialectoBot</h1>
      <p>Welcome to DialectoBot! Enter any Spanish 
        sentence to detect whether it's Mexican (MX) 
        or Spain (ES) Spanish</p>
      
      {/* Chat message history */}
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            <p>{msg.text}</p>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      
      {isLoading && <p className="loading">Detectando dialecto...</p>}

      {/* Input area that's fixed at bottom like a chat app */}
      <div className="input-area">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="Escribe una frase en español..."
        />
        <button onClick={() => handleSubmit()}>
          Enviar
        </button>
      </div>

    </div>
  )


}

export default App