import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from .gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {}})
        self.from_rep = {"email": "rep@example.com", "name": "Rep"}

    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Introduction",
            body="Hello",
            runtime=self.runtime,
            from_rep=self.from_rep,
        )

    def test_disqualified_prospect_is_blocked_without_sending(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Priya Nair",
            "email": "priya.nair@brightwaveapps.com",
        }
        with patch("gtm_agent.gtm_agent.uuid.uuid4") as uuid4:
            result = self.send(prospect)

        self.assertEqual(result["status"], "blocked")
        self.assertIn("disqualified", result["error"])
        uuid4.assert_not_called()

    def test_qualified_prospect_still_sends(self):
        prospect = {
            "prospect_id": "LEAD-12853",
            "name": "Omar Okafor",
            "email": "omar.okafor@lakesideanalytics.com",
        }
        result = self.send(prospect)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])

    def test_email_fallback_blocks_missing_or_unresolved_prospect_id(self):
        for prospect_id in (None, "LEAD-NOT-FOUND"):
            with self.subTest(prospect_id=prospect_id):
                prospect = {
                    "prospect_id": prospect_id,
                    "name": "Priya Nair",
                    "email": "PRIYA.NAIR@BRIGHTWAVEAPPS.COM",
                }
                with patch("gtm_agent.gtm_agent.uuid.uuid4") as uuid4:
                    result = self.send(prospect)

                self.assertEqual(result["status"], "blocked")
                self.assertIn("disqualified", result["error"])
                uuid4.assert_not_called()

    def test_prompt_requires_reporting_blocked_sends(self):
        self.assertIn("tell the rep the email was not sent because the prospect is disqualified", SYSTEM_PROMPT)
        self.assertIn("never describe a blocked send as successful", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
