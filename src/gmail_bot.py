import base64
import time
import torch
import re
from email.message import EmailMessage
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= CONFIG ================= #

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send"  
]

LOGISTICS_KEYWORDS = {
    "shipcube": 5,
    "order": 2,
    "tracking": 2,
    "shipment": 3,
    "delivery": 2,
    "delay": 2,
    "customs": 2,
    "freight": 2,
    "cargo": 2,
    "warehouse": 2,
    "shipping": 2,
    "container": 2,
    "hub": 1,
    "stuck": 1
}

BLOCKED_PATTERNS = ["noreply", "no-reply", "mailer-daemon", "postmaster"]

VALID_LABELS = {"LOGISTICS", "SHIPCUBE"}

# ========================================== #

class LogisticsEmailBot:

    def __init__(
        self,
        model_path="checkpoints/shipcube_2Dlora_model/checkpoint-450",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        debug=True  # NEW: Enable detailed debugging
    ):
        self.service = None
        self.base_model = base_model
        self.model_path = model_path
        self.debug = debug

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = None
        self.tokenizer = None

        print(f"[INIT] Device: {self.device}")
        print(f"[INIT] Debug mode: {self.debug}")

    # ============ AUTH ============ #

    def authenticate(self):
        print(f"[AUTH] Required scopes: {SCOPES}")
        print("[AUTH] If you've run this before, delete token.json and re-authenticate!")
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        self.service = build("gmail", "v1", credentials=creds)
        print("[AUTH] Gmail authenticated successfully")

    # ============ MODEL ============ #

    def load_model(self):
        print("[MODEL] Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[MODEL] Loading base model")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None
        )

        print("[MODEL] Loading LoRA adapter")
        self.model = PeftModel.from_pretrained(
            base,
            self.model_path,
            dtype=self.dtype
        )
        self.model.eval()
        print("[MODEL] Model loaded and ready")
        
        # FIX #2: Test model after loading
        if self.debug:
            print("[MODEL] Testing generation...")
            test_response = self.generate_response(
                "Test shipment inquiry", 
                "Where is my package?"
            )
            print(f"[MODEL] Test response length: {len(test_response)} chars")

    # ============ HELPERS ============ #

    def extract_email(self, sender):
        """Extract email address from sender string"""
        match = re.search(r"<(.+?)>", sender)
        email = match.group(1) if match else sender.strip()
        
        if self.debug:
            print(f"[DEBUG] Extracted email: '{email}' from '{sender}'")
        
        return email

    def is_valid_sender(self, email):
        """Check if sender is not a blocked pattern"""
        email_lower = email.lower()
        is_valid = not any(b in email_lower for b in BLOCKED_PATTERNS)
        
        if self.debug and not is_valid:
            matched = [b for b in BLOCKED_PATTERNS if b in email_lower]
            print(f"[DEBUG] Blocked sender '{email}' - matched patterns: {matched}")
        
        return is_valid

    def calculate_score(self, text):
        """Calculate logistics relevance score based on keywords"""
        score = 0
        keywords = []
        text_lower = text.lower()
        for keyword, weight in LOGISTICS_KEYWORDS.items():
            if keyword in text_lower:
                score += weight
                keywords.append(keyword)
        return score, keywords

    # ============ FILTER CORE ============ #

    def is_logistics_related(self, subject, body, sender, labels):
        """Determine if email is logistics-related"""
        sender_email = self.extract_email(sender)

        if not self.is_valid_sender(sender_email):
            print(f"[FILTER] ❌ Blocked sender: {sender_email}")
            return False, 0, []

        combined = f"{subject} {body}".lower()

        # Check for Shipcube mention
        shipcube_present = "shipcube" in combined

        # Check for relevant labels
        label_present = any(lbl.upper() in VALID_LABELS for lbl in labels)

        # Calculate keyword scores
        subject_score, subject_kw = self.calculate_score(subject)
        body_score, body_kw = self.calculate_score(body)

        # Weight subject more heavily
        total_score = subject_score * 2 + body_score
        keywords = list(set(subject_kw + body_kw))

        # Pass if score >= 2
        logistics_score_pass = total_score >= 2

        # FIX #3: Enhanced debug output
        if self.debug:
            print(f"\n[FILTER] === Email Analysis ===")
            print(f"[FILTER] Subject: '{subject}'")
            print(f"[FILTER] Body preview: '{body[:150]}...'")
            print(f"[FILTER] Sender: {sender_email}")
            print(f"[FILTER] Labels: {labels}")
            print(f"[FILTER] ---")
            print(f"[FILTER] Shipcube present: {shipcube_present}")
            print(f"[FILTER] Valid label present: {label_present}")
            print(f"[FILTER] Subject score: {subject_score} (keywords: {subject_kw})")
            print(f"[FILTER] Body score: {body_score} (keywords: {body_kw})")
            print(f"[FILTER] Total score: {total_score} (threshold: 2)")
            print(f"[FILTER] Score passes: {logistics_score_pass}")
            print(f"[FILTER] ===================\n")

        if shipcube_present or label_present or logistics_score_pass:
            print(f"[FILTER] ✅ PASS - Logistics email detected")
            return True, total_score, keywords

        print(f"[FILTER] ❌ FAIL - Not logistics-related")
        return False, total_score, keywords

    # ============ GMAIL ============ #

    def get_unread_emails(self):
        """Fetch unread emails"""
        query = "is:unread"
        try:
            res = self.service.users().messages().list(
                userId="me", q=query, maxResults=10
            ).execute()
            messages = res.get("messages", [])
            print(f"[GMAIL] Found {len(messages)} unread messages")
            return messages
        except Exception as e:
            print(f"[ERROR] Failed to fetch emails: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_email_content(self, msg_id):
        """Retrieve full email content"""
        msg = self.service.users().messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()

        headers = msg["payload"]["headers"]
        
        def get_header(key):
            return next((h["value"] for h in headers if h["name"].lower() == key.lower()), "")

        body = self.extract_body(msg["payload"])
        
        email_data = {
            "id": msg_id,
            "sender": get_header("from"),
            "subject": get_header("subject"),
            "message_id": get_header("message-id"),
            "thread_id": msg["threadId"],
            "labels": msg.get("labelIds", []),
            "body": body
        }
        
        if self.debug:
            print(f"\n[EMAIL] === Retrieved Email ===")
            print(f"[EMAIL] From: {email_data['sender']}")
            print(f"[EMAIL] Subject: {email_data['subject']}")
            print(f"[EMAIL] Body length: {len(body)} chars")
            print(f"[EMAIL] Labels: {email_data['labels']}")
            print(f"[EMAIL] ======================\n")
        
        return email_data

    def extract_body(self, payload):
        """Extract text body from email payload"""
        def get_text_from_parts(parts):
            for part in parts:
                if part["mimeType"] == "text/plain":
                    if "data" in part["body"]:
                        return base64.urlsafe_b64decode(part["body"]["data"]).decode(errors="ignore")
                if "parts" in part:
                    text = get_text_from_parts(part["parts"])
                    if text:
                        return text
            return ""
        
        # Check for multipart
        if "parts" in payload:
            text = get_text_from_parts(payload["parts"])
            if text:
                return text
        
        # Check for simple body
        if "body" in payload and "data" in payload["body"]:
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode(errors="ignore")
        
        return ""

    # ============ AI ============ #

    def generate_response(self, subject, body):
        """Generate AI response to email"""
        prompt = f"""<|system|>
You are a professional Shipcube logistics support assistant. Provide helpful, concise responses to customer inquiries about shipments, tracking, and logistics.
</s>
<|user|>
Subject: {subject}

{body[:500]}
</s>
<|assistant|>
"""

        print("[AI] Generating response...")
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract response after assistant tag
            if "<|assistant|>" in decoded:
                response = decoded.split("<|assistant|>")[-1].strip()
            else:
                response = decoded.split("</s>")[-1].strip()

            # Fallback if response is too short
            if len(response) < 50:
                response = (
                    "Thank you for contacting Shipcube. We have received your inquiry regarding your shipment "
                    "and our team is reviewing the details. We will provide you with a detailed update shortly.\n\n"
                    "If you have any urgent concerns, please don't hesitate to reach out."
                )

            # Ensure proper closing
            if "shipcube" not in response.lower() and "best regards" not in response.lower():
                response += "\n\nBest regards,\nShipcube Logistics Team"

            print(f"[AI] ✅ Generated {len(response)} chars")
            if self.debug:
                print(f"[AI] Response preview: {response[:200]}...")
            
            return response
            
        except Exception as e:
            print(f"[ERROR] ❌ AI generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Return a safe fallback
            return (
                "Thank you for contacting Shipcube. We have received your message "
                "and our support team will respond to you shortly.\n\n"
                "Best regards,\nShipcube Logistics Team"
            )

    # ============ SEND ============ #

    def send_reply(self, email, body):
        """Send email reply"""
        try:
            print(f"[SEND] Preparing reply to: {email['sender']}")
            
            msg = EmailMessage()
            msg.set_content(body)
            msg["To"] = email["sender"]
            msg["Subject"] = f"Re: {email['subject']}" if email['subject'] else "Re: Your inquiry"
            
            if email["message_id"]:
                msg["In-Reply-To"] = email["message_id"]
                msg["References"] = email["message_id"]

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            
            send_body = {"raw": raw}
            if email["thread_id"]:
                send_body["threadId"] = email["thread_id"]
            
            print("[SEND] Calling Gmail API to send...")
            result = self.service.users().messages().send(
                userId="me",
                body=send_body
            ).execute()
            
            print(f"[SEND] ✅ SUCCESS - Message ID: {result.get('id')}")
            return True
            
        except Exception as e:
            print(f"[SEND] ❌ FAILED - Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def mark_as_read(self, msg_id):
        """Mark email as read"""
        try:
            self.service.users().messages().modify(
                userId="me",
                id=msg_id,
                body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            print(f"[MARKED] ✅ Email {msg_id} marked as read")
        except Exception as e:
            print(f"[MARKED] ❌ Failed to mark as read: {e}")
            import traceback
            traceback.print_exc()

    # ============ LOOP ============ #

    def run(self, interval=10, max_iterations=None):
        """Main bot loop"""
        print("=" * 60)
        print("[BOT] 🤖 Logistics Email Bot Started")
        print(f"[BOT] Checking every {interval} seconds")
        print("[BOT] Press Ctrl+C to stop")
        print("=" * 60)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n{'=' * 60}")
                print(f"[LOOP] 🔄 Iteration {iteration}")
                print(f"{'=' * 60}")
                
                messages = self.get_unread_emails()
                
                if not messages:
                    print("[LOOP] 📭 No unread messages")
                else:
                    for idx, m in enumerate(messages, 1):
                        print(f"\n[LOOP] Processing email {idx}/{len(messages)}")
                        try:
                            email = self.get_email_content(m["id"])

                            ok, score, kw = self.is_logistics_related(
                                email["subject"],
                                email["body"],
                                email["sender"],
                                email["labels"]
                            )

                            if not ok:
                                print("[SKIP] ⏭️  Not logistics-related, leaving unread")
                                continue

                            print(f"[PROCESS] 📦 Processing logistics email (score: {score}, keywords: {kw})")
                            
                            reply = self.generate_response(
                                email["subject"],
                                email["body"]
                            )

                            if self.send_reply(email, reply):
                                self.mark_as_read(m["id"])
                                print("[DONE] ✅ Successfully replied and marked read")
                            else:
                                print("[FAILED] ❌ Could not send reply - email remains unread")

                        except Exception as e:
                            print(f"[ERROR] ❌ Processing email {m['id']}: {e}")
                            import traceback
                            traceback.print_exc()

                if max_iterations and iteration >= max_iterations:
                    print(f"\n[BOT] 🛑 Reached max iterations ({max_iterations})")
                    break
                
                print(f"\n[LOOP] 💤 Sleeping for {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n[BOT] 🛑 Stopped by user")

# ============ MAIN ============ #

if __name__ == "__main__":
    bot = LogisticsEmailBot(debug=True)  # Enable debug mode
    bot.authenticate()
    bot.load_model()
    bot.run(interval=10)  # Check every 30 seconds