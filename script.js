document.addEventListener('DOMContentLoaded', () => {

    let memberData = [];
    let memberId = getRandomInt(300)+1;
    let totalVideos = 2;
    const micButton = document.getElementById('mic-button');
    const responseInput = document.getElementById('chat-input');
    let recognizing = false;
    let recognition;
    let messages = [];
    const videoElement = document.getElementById('background-video');
    const player = videojs(videoElement);
    let mic_input_text = '';
    const azureKey = 'fvgyP2xTbhnH-Uq5J36NKbGB9FZGwfK1-tT4FuDn3n5PAzFugBHanw=='; 
    document.getElementById('button-visit-1').innerText = '06/'+(getRandomInt(9)+15)+'/2024 \n'+(getRandomInt(9)+8)+':'+getRandomInt(6)+getRandomInt(10)
    document.getElementById('button-visit-2').innerText = '06/'+(getRandomInt(9)+15)+'/2024 \n'+(getRandomInt(9)+8)+':'+getRandomInt(6)+getRandomInt(10)
    document.getElementById('button-visit-3').innerText = '06/'+(getRandomInt(9)+15)+'/2024 \n'+(getRandomInt(9)+8)+':'+getRandomInt(6)+getRandomInt(10)
    var final_message = {
        "General Medical": "Based on your symptoms, I recommend booking an appointment in 24/7. Here are some available timeslots, you can click on the one that suits you best",
        "Primary Care": "Based on your symptoms, I recommend booking an appointment in Primary Care. Here are some available timeslots, you can click on the one that suits you best",
        "Mental Health": "Based on your symptoms, I recommend booking an appointment in Mental Health. Here are some available timeslots, you can click on the one that suits you best",
        "Nutrition": "Based on your symptoms, I recommend booking an appointment in Nutrition. Here are some available timeslots, you can click on the one that suits you best",
        "Dermatology": "Based on your symptoms, I recommend booking an appointment in Dermatology. Here are some available timeslots, you can click on the one that suits you best",
        "Emergency": "This looks like an emergency, please call 911",
        "CCM": "Based on your symptoms, I recommend booking an appointment in CCM. Here are some available timeslots, you can click on the one that suits you best"
    };
    
    document.getElementById('chat-form').addEventListener('submit', function(event) {
        event.preventDefault();
        var input = document.getElementById('chat-input');
        userMessage(input.value)
        input.value = '';
    });

    document.getElementById('chat-input').addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            document.querySelector('.chat-submit').click();
        }
    });


    document.getElementById('start-button').addEventListener('click', () => {
        document.getElementById('popup').style.display = 'none';
        document.getElementById('container').style.display = '';
        addMessageToChat('assistant',JSON.stringify({'text': "Hi, I'm Laia! I'm here to assist you into choosing the correct Teladoc program for your care. Could you share with me what symptoms do you have? ",'isFinal':false}))
        playNewVideo('resources/init_video.mp4');
    })

    document.getElementById('user-button').addEventListener('click', () => {
        let new_display = '';
        if (document.getElementById('info_user').style.display == ''){
            new_display = 'none';
        }
        document.getElementById('info_user').style.display = new_display;
    })

    fetch('data/pii.csv')
        .then(response => response.text())
        .then(text => {
            memberData= parseCSV(text);
            document.getElementById('member_id').textContent = 'Member Id: '+memberId;
            document.getElementById('member_gender').textContent = 'Age: '+memberData[memberId]["age"];
            document.getElementById('member_age').textContent = 'Gender: '+memberData[memberId]["gender"];
        })
        .catch(error => console.error('Error fetching the CSV file:', error));

    function parseCSV(text) {
        const lines = text.split('\n');
        const result = [];
        const headers = lines[0].split(',');
    
        for (let i = 1; i < lines.length; i++) {
            const obj = {};
            const currentline = lines[i].split(',');
    
            for (let j = 0; j < headers.length; j++) {
                obj[headers[j]] = currentline[j];
            }
            result.push(obj);
        }
        return result;
    }
    function userMessage(message){
        if (message.trim() !== '') {
            addMessageToChat('user', message);
        }
        generateResponse();
    }

    function addMessageToChat(type, message) {
        const chatMessages = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', type);
        messageElement.textContent = type=='assistant' ? JSON.parse(message)['text']:message;
        messageElement.style.width = (message.length*8.15) + 'px';
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        messages.push({"role":type,"content":message});
    }


    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-EN';

        recognition.onstart = function() {
            recognizing = true;
            document.getElementById('mic-img').src = 'resources/no_voice.png'
            document.getElementById('mic-img').style.width = '25px';
        };

        recognition.onend = function() {
            recognizing = false;
            //userMessage(mic_input_text)
            document.getElementById('mic-img').src = 'resources/on_voice.png'
            document.getElementById('mic-img').style.width = '18px';
            responseInput.value = mic_input_text
        };

        recognition.onresult = function(event) {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            mic_input_text = finalTranscript || interimTranscript;
        };
    }

    micButton.addEventListener('click', function() {
        if (recognizing) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });

    function addSpinner(){
        const chatMessages = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', 'spinner','spinner_chat');
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeSpinner() {
        const chatMessages = document.getElementById('chat-messages');
        const spinnerElements = chatMessages.getElementsByClassName('spinner');
        if (spinnerElements.length > 0) {
            const lastSpinner = spinnerElements[spinnerElements.length - 1];
            chatMessages.removeChild(lastSpinner);
        }
    }


    function generateResponse() {
        const url = `https://laia-backend.azurewebsites.net/api/generate?code=${azureKey}`;
        console.log(messages);

        const requestBody = {
            messages: messages,
            memberId: memberId
        };
        addSpinner();
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.text();
            })
            .then(data => {
                console.log('DATA:', data);
                generateAvatar(data)
            })
            .catch(error => console.error('Error:', error));
    }

    function generateAvatar(data_text) {
        const url = `https://laia-backend.azurewebsites.net/api/avatar?code=${azureKey}`;
        var text = JSON.parse(data_text)['text'];
        if(JSON.parse(data_text)['isFinal'] ){
            var finish_data = JSON.parse(data_text);
            var type=finish_data.text;
            text = final_message[finish_data.text]
            if(text == ''){
                for (var key in final_message) {
                    if (type.includes(key)) {
                        text = final_message[key];
                        break
                    }
                }
            }
        }
        const requestBody = {
            message: text
        };
        console.log(messages[messages.length - 1].content);
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.blob();
            })
            .then(data => {
                var url = URL.createObjectURL(data);
                playNewVideo(url);
                removeSpinner();
                if(JSON.parse(data_text)['isFinal']){
                    data_text = finishMessage(data_text);
                }
                addMessageToChat('assistant', data_text); 
            })
            .catch(error => {
                removeSpinner();
                if(JSON.parse(data_text)['isFinal'] ){
                    data_text = finishMessage(data_text);
                 }
                addMessageToChat('assistant', data_text);
            });
    }

    function finishMessage(data_text){
        var finish_data = JSON.parse(data_text);
        var type=finish_data.text;
        finish_data['text'] = final_message[finish_data.text]
        if(finish_data['text'] == ''){
            for (var key in final_message) {
                if (type.includes(key)) {
                    finish_data['text'] = final_message[key];
                    break
                }
            }
        }
        data_text = JSON.stringify(finish_data);
        document.getElementById('chat-form').style.display = 'none';
        if(type != 'Emergency'){
            document.getElementById('schedule').style.display = 'flex';
        }
        return data_text;
    }

    function playNewVideo(videoUrl) {
        player.pause();
        player.src({ type: 'video/mp4', src: videoUrl });
        player.loop(false);
        player.muted(false);
        player.play();
    }

    player.on('ended', () => {
        player.pause();
        player.src({ type: 'video/mp4', src: 'resources/wait_video_'+getRandomInt(totalVideos)+'.mp4' });
        player.loop(true);
        player.muted(true);
        player.play();
    });

    function getRandomInt(max) {
        return Math.floor(Math.random() * max);
      }

    
    const buttonIds = ['button-visit-1', 'button-visit-2', 'button-visit-3'];
    function schedulingClick(event) {
        let textoBoton = event.target.innerText.split('\n');
        let fecha = textoBoton[0];
        let hora = textoBoton[1];

        let message = `Your appointment has been confirmed, please meet your doctor on ${fecha} at ${hora}.`;

        addMessageToChat("assistant",JSON.stringify({text:message}))
        document.getElementById('schedule').style.display = 'none';
    }

    // Lista de IDs de los botones

    // Asigna el texto inicial y el manejador de eventos a cada botón
    buttonIds.forEach(id => {
        document.getElementById(id).addEventListener('click', schedulingClick);
    });
});
