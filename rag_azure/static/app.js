const chatEl = document.getElementById("chat");
const form = document.getElementById("chat-form");
const msgInput = document.getElementById("msg");

let session_id = null;

function appendMessage(role, text){
  const wrapper = document.createElement("div");
  wrapper.classList.add("message");
  wrapper.classList.add(role==="user"?"user":"assistant");
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.innerText = text;
  wrapper.appendChild(bubble);
  chatEl.appendChild(wrapper);
  chatEl.scrollTop = chatEl.scrollHeight;
}

form.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const message = msgInput.value.trim();
  if(!message) return;
  appendMessage("user", message);

  msgInput.value = "";
  msgInput.disabled = true;
  form.querySelector("button").disabled = true;

  try {
    const res = await fetch("/api/chat",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({ message, session_id, top_k:5 })
    });
    const data = await res.json();
    if(data.error) appendMessage("assistant", `Errore: ${data.error}`);
    else {
      appendMessage("assistant", data.answer || "Nessuna risposta");
      session_id = data.session_id; // salva sessione
    }
  } catch(err){
    appendMessage("assistant","Errore di connessione al server.");
    console.error(err);
  } finally{
    msgInput.disabled = false;
    form.querySelector("button").disabled = false;
    msgInput.focus();
  }
});
