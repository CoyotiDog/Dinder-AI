//STUFF
let redditSubs = ["code","javscript","all"];








// ======= CORE LLM =======
let vocab={}, revVocab={}, wordIndex=0;
let W=[], contextLength=3;
let chatHistory=[];
let lastWordsQueue=[], maxRepeat=20;

function tokenize(text){ return text.toLowerCase().match(/\b[\w']+|[.!?]/g)||[]; }

function ensureWord(word){
  if(!(word in vocab)){
    vocab[word]=wordIndex; revVocab[wordIndex]=word; wordIndex++;
    for(let i=0;i<W.length;i++) W[i].push(Math.random()*0.01);
    W.push(Array(wordIndex).fill(0).map(()=>Math.random()*0.01));
  }
  return vocab[word];
}

// ======= COMBINED TRAIN FUNCTION =======
function trainWordLevel(text, times){
    if(!document.getElementById("learningToggle").checked) return;

    const blacklist=["badword1","badword2"];
    text = text.split(/\s+/).filter(w=>!blacklist.includes(w.toLowerCase())).join(" ");

    let words = text.split(/\s+/);
    let repeatCount=0;
    for(let w of words){ if(lastWordsQueue.includes(w)) repeatCount++; }
    if(repeatCount>maxRepeat){ appendChat("AI","(Skipped training: repeated words detected)"); return; }
    lastWordsQueue.push(...words);
    if(lastWordsQueue.length>50) lastWordsQueue.splice(0,lastWordsQueue.length-50);

    const lr=parseFloat(document.getElementById("learningRate").value);
    for(let t=0; t<words.length-1; t++){
        const x=ensureWord(words[t]);
        const y=ensureWord(words[t+1]);
        for(let n=0; n<times; n++){
            let logits = W[x];
            let probs = logits.map(v=>Math.exp(v));
            let sum = probs.reduce((a,b)=>a+b,0);
            probs = probs.map(p=>p/sum);
            for(let j=0;j<probs.length;j++){
                let grad = probs[j]; if(j===y) grad-=1;
                W[x][j]-=lr*grad;
            }
        }
    }
}

// ======= CONTEXT-AWARE GENERATION =======
function generateWordLevel(maxWords=50, userContext="") {
  if(wordIndex===0) return ["..."];
  
  let contextWords = tokenize(userContext).map(w => vocab[w]).filter(v => v !== undefined);
  let context=[], result=[];
  let idx = contextWords.length>0 ? contextWords[Math.floor(Math.random()*contextWords.length)] : Math.floor(Math.random()*wordIndex);
  context.push(idx); result.push(revVocab[idx] || "...");

  for(let t=1; t<maxWords; t++) {
    let logits = W[idx]||Array(wordIndex).fill(0.01);
    let temp = parseFloat(document.getElementById("temperature").value);
    
    let probs = logits.map(v => Math.exp(v/temp));

    if(contextWords.length>0){
      for(let cw of contextWords){
        if(probs[cw] !== undefined) probs[cw] *= 1.5;
      }
    }

    let sum = probs.reduce((a,b)=>a+b,0);
    probs = probs.map(p=>p/sum);

    let topIndices = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]).slice(0,3).map(x=>x[1]);
    let nextIdx = topIndices[Math.floor(Math.random()*topIndices.length)];
    result.push(revVocab[nextIdx] || "...");
    context.push(nextIdx); if(context.length>contextLength) context.shift();
    idx = nextIdx;
  }

  return result;
}

// ======= CHAT =======
function appendChat(sender,text){
  const out=document.getElementById("chatOutput");
  out.textContent+=sender+": "+text+"\n";
  out.scrollTop=out.scrollHeight;
}

let conversationMetrics={totalResponses:0,garbledResponses:0};
const oldAppendChat=appendChat;
appendChat=function(sender,text){
  if(sender==="AI") conversationMetrics.totalResponses++;
  if(sender==="AI" && text.includes("garbled")) conversationMetrics.garbledResponses++;
  oldAppendChat(sender,text);
}
setInterval(()=>{ console.log("Conversation Metrics:",conversationMetrics); },30000);

function clearOutput(){ document.getElementById("chatOutput").textContent=""; }

function sendMessage(){
  let text=document.getElementById("chatText").value;
  if(!text) return;
  chatHistory.push("User: "+text);
  appendChat("User",text);
  trainWordLevel(text,10);
  document.getElementById("chatText").value="";
  
  generateAIResponseDelayed(text);
}

function generateAIResponseDelayed(userText){
  appendChat("AI","typing...");
  let delay=500+Math.random()*1000;
  setTimeout(()=>{
    let out=document.getElementById("chatOutput");
    out.textContent=out.textContent.replace(/AI: typing...\n$/,"");
    let aiWords=generateWordLevel(50, userText);
    if(aiWords.length===0) aiWords=["..."];
    if(checkGarbled(aiWords)) appendChat("AI","(Warning: garbled output detected)");
    let aiResp=aiWords.join(" ");
    appendChat("AI",aiResp);
    chatHistory.push("AI: "+aiResp);
  },delay);
}

function checkGarbled(words){
  let unknownCount=words.filter(w=>!(w in vocab)).length;
  return unknownCount/words.length>0.5;
}

// ======= EXTRA FEATURES =======
let extraUserName="User";
setTimeout(()=>{ let n=prompt("Enter your name:","User"); if(n) extraUserName=n; },500);

const extraSites=[
  "https://en.wikipedia.org/api/rest_v1/page/random/summary",
  "https://api.publicapis.org/entries"
];
async function autoMultiSiteTrain(){ 
  for(let url of extraSites){ 
    try{ 
      const res=await fetch(url); 
      const data=await res.json(); 
      let text=data.extract||JSON.stringify(data); 
      if(text) trainWordLevel(text,100); 
    }catch(e){console.log("Fetch failed:",url);} 
  } 
}
setInterval(autoMultiSiteTrain,5000);


async function autoRedditTrain(){
  for(let sub of redditSubs){
    try{
      const res = await fetch(`https://www.reddit.com/r/${sub}/hot.json?limit=5`);
      const data = await res.json();
      if(data && data.data && data.data.children){
        data.data.children.forEach(post=>{
          let text = post.data.title + " " + (post.data.selftext || "");
          if(text) trainWordLevel(text,50);
        });
      }
    }catch(e){console.log("Reddit auto-train failed for /r/"+sub,e);}
  }
}
setInterval(autoRedditTrain,15000);

// ======= CONTROLS =======
function train(times){
  let text=document.getElementById("trainData").value;
  if(!text) return;
  trainWordLevel(text,times);
  appendChat("AI",`Trained ${times}x on your input.`);
}
function resetWeights(){ W=Array(wordIndex).fill(0).map(()=>Array(wordIndex).fill(0).map(()=>Math.random()*0.01)); appendChat("AI","Weights reset."); }
function saveSession(){ localStorage.setItem("babyLLMWordWeights",JSON.stringify(W)); localStorage.setItem("babyLLMVocab",JSON.stringify(vocab)); appendChat("AI","Session saved."); }
function loadSession(){
  let w=localStorage.getItem("babyLLMWordWeights"); let v=localStorage.getItem("babyLLMVocab");
  if(w && v){ W=JSON.parse(w); vocab=JSON.parse(v); revVocab={}; for(let k in vocab) revVocab[vocab[k]]=k; wordIndex=Object.keys(vocab).length; appendChat("AI","Session loaded."); } 
  else appendChat("AI","No saved session.");
}
function toggleSettings(){ const panel=document.getElementById("settingsPanel"); panel.style.display = panel.style.display==="none"?"block":"none"; }

// ======= VISUALIZATION =======
function drawWeights(){
  if(!document.getElementById("weightToggle").checked) return;
  const canvas=document.getElementById("weightCanvas"); const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let i=0;i<W.length;i++){
    for(let j=0;j<W[i].length;j++){
      const intensity=Math.min(255,Math.max(0,Math.floor(W[i][j]*1000)));
      ctx.fillStyle=`rgb(${intensity},${intensity},0)`;
      ctx.fillRect(j*canvas.width/W[i].length, i*canvas.height/W.length, canvas.width/W[i].length, canvas.height/W.length);
    }
  }
}
function drawHeatmap(){
  if(!document.getElementById("attentionToggle").checked) return;
  const canvas=document.getElementById("heatmapCanvas"); const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let i=0;i<W.length;i++){
    const avg=W[i].reduce((a,b)=>a+b,0)/W[i].length;
    const intensity=Math.min(255,Math.max(0,Math.floor(avg*1000)));
    ctx.fillStyle=`rgb(0,${intensity},0)`;
    ctx.fillRect(i*canvas.width/W.length,0,canvas.width/W.length,canvas.height);
  }
}
setInterval(()=>{ drawWeights(); drawHeatmap(); },1000);

document.getElementById("temperature").addEventListener("input", e=>document.getElementById("tempVal").textContent=e.target.value);

document.addEventListener('keydown',e=>{ 
  if(e.ctrlKey && e.key.toLowerCase()==='d') {
    darkMode=!darkMode; 
    const c=darkMode?{bg:"#111",fg:"#0f0"}:{bg:"#fff",fg:"#000"};
    document.body.style.background=c.bg; document.body.style.color=c.fg;
    document.querySelectorAll('textarea,button,select,input[type=range],input[type=color]').forEach(el=>{
      el.style.background=c.bg; el.style.color=c.fg; el.style.border=`1px solid ${c.fg}`;
    });
  }
});

