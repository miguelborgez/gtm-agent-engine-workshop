import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

import gtm_agent.gtm_agent as scoring


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "description": "Data platform",
            "required_tech_stack": ["Snowflake", "dbt", "Kafka"],
            "min_annual_revenue": 50_000_000,
        }
        self.prospect = {
            "annual_revenue": 60_000_000,
            "tech_stack": ["Snowflake", "dbt"],
        }
        self.llm_result = scoring.ProspectScore(
            score=88,
            justification="The prospect has a strong fit.",
            rubric_breakdown={
                "revenue_fit": 72,
                "tech_stack_match": 68,
                "segment_fit": 80,
            },
        )

    def score(self, prospect=None, offering=None):
        with patch.object(scoring, "_scoring_llm") as mock_llm:
            mock_llm.invoke.side_effect = lambda messages: self.llm_result.model_copy(deep=True)
            return scoring.score_prospect.invoke({
                "prospect_profile": prospect or self.prospect,
                "offering": offering or self.offering,
            })

    def test_identical_inputs_return_identical_scores(self):
        first = self.score()
        second = self.score()

        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["rubric_breakdown"], second["rubric_breakdown"])

    def test_score_is_weighted_sum_of_rubric_components(self):
        result = self.score()
        components = result["rubric_breakdown"]
        self.assertEqual(1, sum(scoring.SCORING_WEIGHTS.values()))
        expected = round(sum(
            components[name] * weight
            for name, weight in scoring.SCORING_WEIGHTS.items()
        ), 1)

        self.assertEqual(result["score"], expected)
        self.assertAlmostEqual(100 * 2 / 3, components["tech_stack_match"])
        self.assertEqual(100, components["revenue_fit"])

    def test_empty_required_tech_stack_is_full_match(self):
        offering = {**self.offering, "required_tech_stack": []}

        result = self.score(offering=offering)

        self.assertEqual(100, result["rubric_breakdown"]["tech_stack_match"])

    def test_revenue_below_threshold_has_zero_fit(self):
        prospect = {**self.prospect, "annual_revenue": 49_999_999}

        result = self.score(prospect=prospect)

        self.assertEqual(0, result["rubric_breakdown"]["revenue_fit"])

    def test_revenue_at_threshold_has_full_fit(self):
        prospect = {**self.prospect, "annual_revenue": 50_000_000}

        result = self.score(prospect=prospect)

        self.assertEqual(100, result["rubric_breakdown"]["revenue_fit"])

    def test_missing_required_tech_stack_is_invalid(self):
        offering = {key: value for key, value in self.offering.items() if key != "required_tech_stack"}

        result = scoring.score_prospect.invoke({
            "prospect_profile": self.prospect,
            "offering": offering,
        })

        self.assertEqual({"score": None, "error": "Cannot score without a valid offering."}, result)


if __name__ == "__main__":
    unittest.main()
