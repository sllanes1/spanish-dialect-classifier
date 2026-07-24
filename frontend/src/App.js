import React, {useState, useRef, useEffect} from 'react'
import './App.css';
import logo from './Dialectobotlogo.png'

function App() {
  // Track what user is currently typing
  const [inputText, setInputText] = useState("")
  // Full chat history from user and bot
  const [messages, setMessages] = useState([])
  // Controls "Detectando dialecto..." state while waiting for API
  const [isLoading, setIsLoading] = useState(false)
  // null = not logged in, string = logged in with JWT token
  const [token, setToken] = useState(null)
  // State of the username text field
  const [username, setUsername] = useState("")
  // State of the password text field
  const [password, setPassword] = useState("")
  // Error state if user DNE
  const [error, setError] = useState("")
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

  async function handleLogin(){
    // send user and pass to /login via POST
    const response = await fetch("http://127.0.0.1:8000/login", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({username, password})
    })
    const data = await response.json()

    // if successful save the token with setToken
    if (response.ok) {
      setToken(data.token)
    } else {
      setError(data.detail || "Something went wrong, try again!")
    }
  }

  async function handleSignup(){
    // send user and pass to /signup via POST
    const response = await fetch("http://127.0.0.1:8000/signup", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({username, password})
    })
    const data = await response.json()

    if (response.ok) {
      // call /login to get the token
      const loginResponse = await fetch("http://127.0.0.1:8000/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username, password})
      })
      const loginData = await loginResponse.json()
      setToken(loginData.token)
    } else {
      setError(data.detail || "Something went wrong, try again!")
    }
  }
  
  return( 
    <div className="app">
      <img src={logo} alt="DialectoBot" className="logo" />
      <p>Welcome to DialectoBot! Enter any Spanish 
        sentence to detect whether it's Mexican (MX) 
        or Spain (ES) Spanish</p>
      
      {!token ? (
        // Login form goes here
        <div className="login-form">
          <h2> Your Account </h2>
          <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="username"
            />
          <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="password"
            />

          {error && <p className="error">{error}</p>}

          <button onClick={() => handleLogin()}>
            Login  
          </button>
          <button onClick={() => handleSignup()}>
            SignUp
          </button>
        </div>
      ) : (
        // Chat interface goes here
        <>
          <div className="chat-window">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <p>{msg.text}</p>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
          
          {isLoading && <p className="loading">Detectando dialecto...</p>}

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
        </>
      )}

    </div>
  )

}

export default App