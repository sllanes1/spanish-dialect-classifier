import React, {useState} from 'react'
import './App.css';

function App() {
  const [inputText, setInputText] = useState("")
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  return( 
    <div className="app">
      <h1>DialectoBot</h1>
      <p>Welcome to DialectoBot! Enter any Spanish 
        sentence to detect whether it's Mexican (MX) 
        or Spain (ES) Spanish</p>
      
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            <p>{msg.text}</p>
          </div>
        ))}
      </div>



    </div>
  )


}

export default App