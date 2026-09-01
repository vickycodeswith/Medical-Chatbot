$(document).ready(function () {

  let symptoms = JSON.parse(window.symptoms || "[]");

  const input = $("#message-text");
  const sendBtn = $("#send");
  const startOverBtn = $("#start-over");
  const dataList = $("#symptoms-list");
  const chat = $("#conversation");

  let isProcessing = false;


  // ============================================================
  // UTILITIES
  // ============================================================

  function scrollToBottom() {
    if (chat.length) {
      chat.stop().animate(
        {
          scrollTop: chat[0].scrollHeight
        },
        250
      );
    }
  }


  function setInputState(disabled) {
    input.prop("disabled", disabled);

    if (disabled) {
      sendBtn.css("opacity", "0.5");
      sendBtn.css("pointer-events", "none");
    } else {
      sendBtn.css("opacity", "1");
      sendBtn.css("pointer-events", "auto");
    }
  }


  function showTypingIndicator() {

    if ($("#typing-indicator").length) {
      return;
    }

    chat.append(`
      <div class="row message-body" id="typing-indicator">
        <div class="col-sm-12 message-main-receiver">
          <div class="receiver">
            <div class="message-text">
              <span class="typing-text">Meddy is typing</span>
              <span class="typing-dots">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    `);

    scrollToBottom();
  }


  function removeTypingIndicator() {
    $("#typing-indicator").remove();
  }


  function escapeHtml(text) {
    return $("<div>").text(text).html();
  }


  function showErrorMessage() {

    $.fn.appendBotMessage(
      "Sorry, I couldn't connect to the medical prediction service. Please try again."
    );
  }


  // ============================================================
  // SYMPTOM SUGGESTIONS
  // ============================================================

  input.on("input", function () {

    const value = $(this).val().trim();

    dataList.empty();

    if (value.length <= 1) {
      $(".symptoms-list-container").slideUp(120);
      return;
    }

    const suggestions = $.fn.getSuggestedSymptoms(value);

    if (suggestions.length === 0) {
      $(".symptoms-list-container").slideUp(120);
      return;
    }

    suggestions.forEach(function (symptom) {

      const li = document.createElement("li");

      li.textContent = symptom;

      dataList.append(li);
    });

    $(".symptoms-list-container").slideDown(120);
  });


  // ============================================================
  // SUGGESTION CLICK
  // ============================================================

  dataList.on("click", "li", function () {

    input.val($(this).text());

    $(".symptoms-list-container").slideUp(120);

    input.focus();
  });


  // ============================================================
  // HIDE SUGGESTIONS
  // ============================================================

  input.on("blur", function () {

    setTimeout(function () {
      $(".symptoms-list-container").slideUp(120);
    }, 150);

  });


  // ============================================================
  // SEND BUTTON
  // ============================================================

  sendBtn.on("click", function (event) {

    event.preventDefault();

    $.fn.handleUserMessage();
  });


  // ============================================================
  // ENTER KEY
  // ============================================================

  input.on("keydown", function (event) {

    if (event.key === "Enter") {

      event.preventDefault();

      $.fn.handleUserMessage();
    }

  });


  // ============================================================
  // HANDLE USER MESSAGE
  // ============================================================

  $.fn.handleUserMessage = function () {

    if (isProcessing) {
      return;
    }

    const text = input.val().trim();

    if (!text) {
      return;
    }


    // Hide suggestions
    $(".symptoms-list-container").slideUp(120);


    // IMPORTANT:
    // Show user's message immediately
    $.fn.appendUserMessage(text);


    // Clear input
    input.val("");


    // Send text to Flask
    $.fn.getPredictedSymptom(text);

  };


  // ============================================================
  // APPEND USER MESSAGE
  // ============================================================

  $.fn.appendUserMessage = function (text) {

    chat.append(`
      <div class="row message-body">
        <div class="col-sm-12 message-main-sender">
          <div class="sender">
            <div class="message-text">
              ${escapeHtml(text)}
            </div>
          </div>
        </div>
      </div>
    `);

    scrollToBottom();
  };


  // ============================================================
  // APPEND BOT MESSAGE
  // ============================================================

  $.fn.appendBotMessage = function (text) {

    chat.append(`
      <div class="row message-body">
        <div class="col-sm-12 message-main-receiver">
          <div class="receiver">
            <div class="message-text">
              ${text}
            </div>
          </div>
        </div>
      </div>
    `);

    scrollToBottom();
  };


  // ============================================================
  // SEND REQUEST TO FLASK
  // ============================================================

  $.fn.getPredictedSymptom = function (text) {

    isProcessing = true;

    setInputState(true);

    showTypingIndicator();


    $.ajax({

      url: "/symptom",

      type: "POST",

      data: JSON.stringify({
        sentence: text
      }),

      contentType: "application/json; charset=utf-8",

      dataType: "json",


      success: function (response) {

        removeTypingIndicator();

        $.fn.appendBotMessage(response);
      },


      error: function (xhr, status, error) {

        console.error(
          "Chatbot request failed:",
          status,
          error
        );

        console.error(
          "Server response:",
          xhr.responseText
        );

        removeTypingIndicator();

        showErrorMessage();
      },


      complete: function () {

        isProcessing = false;

        setInputState(false);

        input.focus();

        scrollToBottom();
      }

    });

  };


  // ============================================================
  // START OVER
  // ============================================================

  startOverBtn.on("click", function () {

    if (isProcessing) {
      return;
    }

    $.ajax({

      url: "/symptom",

      type: "POST",

      data: JSON.stringify({
        sentence: "done"
      }),

      contentType: "application/json; charset=utf-8",

      dataType: "json"

    });


    removeTypingIndicator();

    chat.empty();


    const welcomeMessage = `
      Welcome! I'm Medical Chatbot, but you can call me Meddy.
      What symptoms are you currently experiencing?

      <br><br>

      When you've entered all of your symptoms, please write
      <b>Done</b>.

      <br><br>

      Make sure you enter as many symptoms as possible so the
      prediction can be as accurate as possible.
    `;


    chat.append(`
      <div class="row message-body">
        <div class="col-sm-12 message-main-receiver">
          <div class="receiver">
            <div class="message-text">
              ${welcomeMessage}
            </div>
          </div>
        </div>
      </div>
    `);


    input.val("");

    input.prop("disabled", false);

    input.focus();

    scrollToBottom();
  });


  // ============================================================
  // SEARCH SYMPTOMS
  // ============================================================

  $.fn.getSuggestedSymptoms = function (value) {

    const result = [];

    const search = value.toLowerCase().trim();


    $.each(symptoms, function (index, symptom) {

      if (
        symptom.toLowerCase().includes(search)
      ) {

        result.push(symptom);
      }

    });


    return result.slice(0, 5);
  };


  // ============================================================
  // INITIALIZE
  // ============================================================

  input.focus();

  scrollToBottom();

});
