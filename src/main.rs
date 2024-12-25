use actix::prelude::*;
use jemini::{GeminiError, JeminiClient};
use log::{debug, error, info};
use rand::seq::SliceRandom;
use std::{collections::HashMap, env, io::{self, Write}, time::Instant};

/// Define feedback (Good or Needs Refinement)
#[derive(Debug, Clone, Copy, PartialEq, MessageResponse)]
enum Feedback {
    Good,
    NeedsRefinement,
}

/// Registers the LLM actor's name and [Addr] with the [Coordinator].
#[derive(Message)]
#[rtype(result = "bool")]
struct Register {
    name: String,
    actor: Addr<LlmActor>
}

/// Sent to the [Coordinator] or an LLM actor to request an answer.
#[derive(Message)]
#[rtype(result = "bool")]
struct AskQuestion(String);

/// Send as the answer to a question posed in [AskQuestion].
#[derive(Message)]
#[rtype(result = "bool")]
struct AnswerQuestion(String);

// Define the message types
#[derive(Message)]
#[rtype(result = "bool")]
struct AnswerReadinessRequest;

#[derive(Message)]
#[rtype(result = "String")]
struct GetAnswer;

#[derive(Debug, Message)]
#[rtype(result = "bool")]
struct EvaluateAnswer {
    question: String,
    answer: String
}

#[derive(Message)]
#[rtype(result = "bool")]
struct AnswerEvaluation {
    name: String,
    evaluation: Feedback,
    reasoning: String
}

#[derive(Message)]
#[rtype(result = "bool")]
struct RefineAnswer {
    question: String,
    answer: String
}

#[derive(Message)]
#[rtype(result = "bool")]
struct AnswerRefinement(String);

#[derive(Message)]
#[rtype(result = "bool")]
struct Reset;

// LLM actor that interacts with LLM API
struct LlmActor {
    name: String,
    domain: String,
    tuning: String,
}

impl Actor for LlmActor {
    type Context = Context<Self>;
}

async fn call_gemini(prompt: String) -> Result<String, GeminiError> {
    let client = JeminiClient::new()?;
    let response = client.text_only(prompt.as_str()).await?;
    Ok(response.most_recent().expect(format!("{} should return an answer", prompt).as_str()).to_owned())
}

// LLM Actor Message Handlers
impl Handler<AskQuestion> for LlmActor {
    type Result = bool;

    fn handle(&mut self, msg: AskQuestion, _: &mut Self::Context) -> Self::Result {
        debug!("LLM actor {} received AskQuestion: {}", self.name, msg.0);

        let prompt = format!("Please answer the following question without referring to yourself as a language model:\n\n{}", msg.0);
        let execution = async move {
            let response = call_gemini(prompt).await.expect("expect successful response");
            Coordinator::from_registry().do_send(AnswerQuestion(response));
        };

        Arbiter::current().spawn(execution);
        true
    }
}

impl Handler<EvaluateAnswer> for LlmActor {
    type Result = bool;

    fn handle(&mut self, msg: EvaluateAnswer, _: &mut Self::Context) -> Self::Result {
        let name = self.name.clone();
        let prompt = format!(r"
---
Question: {}
---
Answer: {}
---
Your Instructions:
You are part of a team of LLMs that were given the above question to answer by consensus. The first model chosen answered with the answer above. You need to evaluate this answer based on your knowledge domain of {}. The only answers you may provide are Good and NeedsRefinement.

Consider how the answer might indirectly or tangentially relate to the domain. A direct connection is not required. Focus on how the answer could enable, inspire, or be used in activities related to the domain. Specifically, you should consider aspects like:{}

The most important part of choosing your answer is whether the question is related to your domain at all. If it is not, then you should answer exactly Good since you are not qualified to evaluate the answer. Otherwise, if you think this was a good answer, respond with exactly Good. If you think this was a bad answer, respond with exactly NeedsRefinement. Additionally, you must also provide reasoning for why you think this answer is Good or NeedsRefinement answer by putting that reasoning on a new line.
---
Examples:

Question: What's a good beginner programming language?
Answer: Python
Your domain: art and imagination
Evaluation: Good
Reasoning: This isn't related to your domain.

Question: How can I make my software easier to update?
Answer: Decoupling
Your domain: technical rigor
Evaluation: NeedsRefinement
Reasoning: Decoupling and high cohesion are only one aspect of maintainable software, and the answer doesn't go into enough detail.", msg.question, msg.answer, self.domain, self.tuning).replace("\"", "");
        let execution = async {
            let result = call_gemini(prompt).await.expect("EvaluateAnswer should produce good response");
            let mut result_parts: Vec<&str> = result.split("\n")
                .filter(|s| !(*s).is_empty())
                .collect();
            let cleaned_result = result_parts[0].replace(" ", "");
            let reasoning = result_parts.split_off(1).join("\n\n");
            Coordinator::from_registry().do_send(AnswerEvaluation{ name: name, evaluation: match cleaned_result.as_str() {
                "Good" => Feedback::Good,
                "NeedsRefinement" => {
                    Feedback::NeedsRefinement
                },
                _ => {
                    error!("Unexpected response from EvaluateAnswer: {}", result);
                    Feedback::NeedsRefinement
                }
            }, reasoning: reasoning});
        };

        Arbiter::current().spawn(execution);
        true
    }
}

impl Handler<RefineAnswer> for LlmActor {
    type Result = bool;

    fn handle(&mut self, msg: RefineAnswer, _: &mut Self::Context) -> Self::Result {
        // Simulate refining the answer by calling OpenAI again with a refinement prompt
        let prompt = format!(r"
---
Question: {}
---
Answer: {}
---
Your Instructions:
A user asked this question, and they received the specified answer. When asked to evaluate this answer, you said it needed refinement. Please refine the answer as necessary for your knowledge domain, {}.

Specifically, keep the following things in mind while refining the answer. They do not need to be included, but they should influence your refinement:{}", msg.question, msg.answer, self.domain, self.tuning).replace("\"", "");

        let execution = async move{
            let response = call_gemini(prompt).await.expect("expect successful response");
            Coordinator::from_registry().do_send(AnswerRefinement(response));
        };

        Arbiter::current().spawn(execution);
        true
    }
}

// Define the Coordinator Actor
#[derive(Default)]
struct Coordinator {
    llm_actors: HashMap<String, Addr<LlmActor>>,
    current_question: Option<String>,
    feedback: HashMap<String, Feedback>,
    answer: Option<String>,
    evaluation_count: u32
}

impl Coordinator {
    fn reset(&mut self) {
        self.current_question = None;
        self.answer = None;
        self.feedback.clear();
        self.evaluation_count = 0;
    }
}

impl Actor for Coordinator {
    type Context = Context<Self>;
}

impl Handler<Register> for Coordinator {
    type Result = bool;

    fn handle(&mut self, msg: Register, _ctx: &mut Self::Context) -> Self::Result {
        self.llm_actors.insert(msg.name.clone(), msg.actor);
        debug!("{} registered with Coordinator.", msg.name);
        true
    }
}

impl Handler<AskQuestion> for Coordinator {
    type Result = bool;

    fn handle(&mut self, msg: AskQuestion, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Received AskQuestion: {}", msg.0);
        self.current_question = Some(msg.0.clone());

        // Select a random LLM actor
        let keys = self.llm_actors.keys().collect::<Vec<&String>>();
        let llm_actor = self.llm_actors.get(keys.choose(&mut rand::thread_rng()).expect("choose() should select a random key").to_owned());

        // Ask the LLM actor for an answer
        match llm_actor {
            Some(addr) =>  {
                addr.do_send(msg);
                true
            },
            None => false,
        }
    }
}

impl Handler<AnswerQuestion> for Coordinator {
    type Result = bool;

    fn handle(&mut self, msg: AnswerQuestion, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Received answer to current question: {}", msg.0);
        self.answer = Some(msg.0.clone());

        debug!("Asking actors to evaluate answer.");
        self.llm_actors.values().for_each(|addr| addr.do_send(EvaluateAnswer{
            question: self.current_question.as_ref().expect("current_question should exist").clone(),
            answer: msg.0.clone()
        }));
        self.evaluation_count += 1;
        true
    }
}

impl Handler<AnswerEvaluation> for Coordinator {
    type Result = bool;

    fn handle(&mut self, msg: AnswerEvaluation, _ctx: &mut Self::Context) -> Self::Result {
        debug!("{} evaluated the answer as {:?}. {}", msg.name, msg.evaluation, msg.reasoning);
        self.feedback.insert(msg.name, msg.evaluation);
        if self.feedback.len() == self.llm_actors.len() {
            if !self.feedback.values().all(|&f| f == Feedback::Good) {
                // Select a random actor that voted NeedsRefinement
                let keys: Vec<String> = self.feedback.clone().into_iter()
                    .filter(|(_, value)| *value == Feedback::NeedsRefinement)
                    .map(|(key, _)| key)
                    .collect();
                let selected_key = keys.choose(&mut rand::thread_rng()).expect("choose() should select a random key").to_owned();
                let llm_actor = self.llm_actors.get(&selected_key);

                let refinement_request = RefineAnswer {
                    question: self.current_question.clone().expect("current_question should exist to get the answer refined"),
                    answer: self.answer.clone().expect("answer should exist to get it refined")
                };
                return match llm_actor {
                    Some(addr) =>  {
                        debug!("Asking {} to refine the answer.", selected_key);
                        addr.do_send(refinement_request);
                        true
                    },
                    None => false,
                }
            }
        }
        true
    }
}

impl Handler<AnswerRefinement> for Coordinator {
    type Result = bool;

    fn handle(&mut self, msg: AnswerRefinement, _ctx: &mut Self::Context) -> Self::Result {
        self.answer = Some(msg.0.clone());
        debug!("Received new answer to current question: {}", msg.0);
        // TODO: Make max count configurable.
        if self.evaluation_count < 5 {
            self.evaluation_count += 1;
            self.feedback.clear();
            debug!("Asking actors to evaluate new answer.");
            self.llm_actors.values().for_each(|addr| addr.do_send(EvaluateAnswer{
                question: self.current_question.as_ref().expect("current_question should exist").clone(),
                answer: msg.0.clone()
            }));
        } else {
            debug!("Evaluated the maximum number of times. Breaking the loop.");
            self.feedback.iter_mut().for_each(|(_, value)| *value = Feedback::Good);
        }
        true
    }
}

impl Handler<AnswerReadinessRequest> for Coordinator {
    type Result = bool;

    fn handle(&mut self, _msg: AnswerReadinessRequest, _ctx: &mut Self::Context) -> Self::Result {
        self.answer.is_some() && 
        !self.feedback.is_empty() && 
        self.feedback.len() == self.llm_actors.len() &&
        self.feedback.values().all(|v| v == &Feedback::Good)
    }
}

impl Handler<GetAnswer> for Coordinator {
    type Result = String;

    fn handle(&mut self, _msg: GetAnswer, _ctx: &mut Self::Context) -> Self::Result {
        if let Some(answer) = &self.answer {
            return answer.to_owned();
        }
        "System error: Requested answer when answer was not ready.".to_string()
    }
}

impl Handler<Reset> for Coordinator {
    type Result = bool;

    fn handle(&mut self, _msg: Reset, _ctx: &mut Self::Context) -> Self::Result {
        self.reset();
        true
    }
}

impl actix::Supervised for Coordinator {}
impl SystemService for Coordinator {}

#[actix::main]
async fn main() {
    env_logger::init();

    if env::var("GEMINI_API_KEY").is_err() {
        error!("No Gemini API key has been set in the GEMINI_API_KEY environment variable. Generate an API key and set it with \"export GEMINI_API_KEY=<your API key>\".");
        return
    }

    Coordinator::from_registry().do_send(Register { 
        name: "High Society".to_string(), 
        actor: LlmActor {
            name: "High Society".to_string(),
            domain: "Society and Culture".to_string(),
            tuning: r"
* Social norms, values, and beliefs
* Historical context and events
* Cultural diversity and traditions
* Social structures and institutions (e.g., family, education, government)
* Impact on human behavior and interactions
* Ethical and moral considerations
* Current events and social issues
* Demographics and population trends
* Communication styles and languages
* Arts, literature, and folklore as reflections of society".to_string()
        }.start()});
    Coordinator::from_registry().do_send(Register { 
        name: "The Technician".to_string(), 
        actor: LlmActor {
            name: "The Technician".to_string(),
            domain: "Technical Detail".to_string(),
            tuning: r"
* Accuracy and precision of information
* Specific measurements, quantities, and units
* Technical specifications and standards
* Detailed procedures and processes
* Scientific principles and theories
* Mathematical formulas and equations
* Logical reasoning and problem-solving
* Causality and cause-and-effect relationships
* Step-by-step explanations and instructions
* Attention to detail and completeness".to_string()
        }.start()});
    Coordinator::from_registry().do_send(Register { 
        name: "Art Boy".to_string(), 
        actor: LlmActor {
            name: "Art Boy".to_string(),
            domain: "Art and Imagination".to_string(),
            tuning: r"
* Creative expression and generation across various mediums (visual, auditory, written, etc.)
* Tools and techniques for artistic creation (digital and traditional)
* Exploration of emotions, ideas, and concepts through art
* Imagination, innovation, and originality
* Aesthetic qualities and principles (e.g., composition, color, form)
* Art history, movements, and styles
* Cultural and social influences on art
* Potential for visualizing data or creating simulations for artistic purposes
* Interactive art and installations
* The role of art in communication and storytelling".to_string()
        }.start()});
    Coordinator::from_registry().do_send(Register { 
        name: "Programming Nerd".to_string(), 
        actor: LlmActor {
            name: "Programming Nerd".to_string(),
            domain: "Computer Science".to_string(),
            tuning: r"
* Algorithms and data structures
* Programming languages and paradigms
* Software engineering principles
* Computer architecture and hardware
* Networking and distributed systems
* Artificial intelligence and machine learning
* Cybersecurity and data privacy
* Computational theory and complexity
* Databases and data management
* Operating systems and system programming".to_string()
        }.start()});

    loop {
        // Get user input
        print!("Enter a question: ");
        io::stdout().flush().expect("stdout should flush"); // Ensure prompt is printed immediately

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("stdin should be able to read a line");
        let question = input.trim().to_string();

        if question == "exit" {
            break;
        }

        // Ask the Coordinator actor
        let question_received = Coordinator::from_registry()
            .send(AskQuestion(question))
            .await
            .expect("should be able to ask question to Coordinator");

        if question_received {
            let mut answer_ready = false;
            let mut timestamp = Instant::now();
            while !answer_ready {
                if timestamp.elapsed().as_millis() < 500 {
                    continue;
                }
                timestamp = Instant::now();
                answer_ready = Coordinator::from_registry()
                    .send(AnswerReadinessRequest)
                    .await
                    .expect("should be able to check answer readiness with the Coordinator");
            }
            let response = Coordinator::from_registry()
                .send(GetAnswer)
                .await
                .expect("should be able to get the answer from the Coordinator");
            info!("Final answer: {}", response);
        }

        Coordinator::from_registry()
            .send(Reset)
            .await
            .expect("Coordinator should reset");
    }
}