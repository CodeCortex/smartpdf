css = '''






<style>
.chat-message {
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    transition: transform 0.2s ease-in-out;
}

.chat-message.user {
    background-color: #3c444f;
    justify-content: flex-end;
    border-radius: 15px;
}

.chat-message.user:hover {
    transform: scale(1.03);
}

/* Bot Message Styles */
.chat-message.bot {
    background-color: #55616b;
    justify-content: flex-start;
    border-radius: 15px;
}

.chat-message.bot:hover {
    transform: scale(1.03);
}

.chat-message .avatar {
    width: 18%;
    margin-right: 15px;
}

.chat-message .avatar img {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
}

.chat-message .message {
    width: 75%;
    padding: 12px 20px;
    color: #fff;
    background-color: #4a5b6a;
    border-radius: 15px;
    word-wrap: break-word;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.chat-message.bot .message {
    background-color: #6c7d87;
}

.chat-message.user .message {
    background-color: #4e5b67;
}

.chat-message.bot {
    flex-direction: row-reverse;  
}

.chat-message.bot .message {
    margin-left: 15px;
    margin-right: 35px;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://img.freepik.com/free-photo/view-graphic-3d-robot_23-2150849173.jpg?t=st=1743760571~exp=1743764171~hmac=722f05bad34875619103f0bb82e39bb4d2f3e00713d64e883e9b2314acb13c31&w=1380" alt="Bot Avatar">
    </div>
    <div class="message">{{MESSAGE}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://images.unsplash.com/photo-1579503841516-e0bd7fca5faa?q=80&w=2960&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="User Avatar">
    </div>    
    <div class="message">{{MESSAGE}}</div>
</div>










'''