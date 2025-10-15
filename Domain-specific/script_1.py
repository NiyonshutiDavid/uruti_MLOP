# Let's expand the dataset with more conversational examples and integrate real startup advice patterns
import json
import pandas as pd
import random
from itertools import combinations

# Load existing data
with open('data/startup_advisory_dataset.json', 'r') as f:
    existing_data = json.load(f)

# Add more conversational and domain-specific examples
additional_conversations = []

# Rwanda-specific startup advice (aligning with Uruti.Rw's Rwandan context)
rwanda_specific = [
    {
        "instruction": "What opportunities exist for startups in Rwanda's economy?",
        "input": "",
        "output": "Rwanda offers excellent opportunities in fintech (mobile payments, digital banking), agritech (supporting smallholder farmers), healthtech (telemedicine, health records), edtech (digital learning platforms), and tourism tech. The government's Vision 2050 and digital transformation agenda provide strong support for tech startups. Focus on solutions that address local challenges while being scalable regionally.",
        "category": "mentorship_needed",
        "text": "### Instruction: What opportunities exist for startups in Rwanda's economy?\n### Response: Rwanda offers excellent opportunities in fintech (mobile payments, digital banking), agritech (supporting smallholder farmers), healthtech (telemedicine, health records), edtech (digital learning platforms), and tourism tech. The government's Vision 2050 and digital transformation agenda provide strong support for tech startups. Focus on solutions that address local challenges while being scalable regionally."
    },
    {
        "instruction": "How can I access funding for my startup in East Africa?",
        "input": "",
        "output": "East Africa has a growing funding ecosystem. Consider: 1) Local VCs like Knife Capital, TLcom Capital, and Novastar Ventures. 2) Development finance institutions like IFC and CDC. 3) Accelerators such as Seedstars, GSMA Ecosystem Accelerator, and local programs. 4) Grants from organizations like USAID, GIZ, and the World Bank. 5) Regional startup competitions and pitch events. Build strong local networks and focus on solving problems relevant to the East African market.",
        "category": "investment_ready",
        "text": "### Instruction: How can I access funding for my startup in East Africa?\n### Response: East Africa has a growing funding ecosystem. Consider: 1) Local VCs like Knife Capital, TLcom Capital, and Novastar Ventures. 2) Development finance institutions like IFC and CDC. 3) Accelerators such as Seedstars, GSMA Ecosystem Accelerator, and local programs. 4) Grants from organizations like USAID, GIZ, and the World Bank. 5) Regional startup competitions and pitch events. Build strong local networks and focus on solving problems relevant to the East African market."
    }
]

# Technical startup questions
technical_qa = [
    {
        "instruction": "Should I build my MVP using no-code tools or custom development?",
        "input": "",
        "output": "For early validation, no-code tools like Bubble, Webflow, or Airtable can be excellent for MVPs. They allow rapid iteration and cost-effective testing. However, consider custom development if: 1) Your product requires complex algorithms, 2) You need specific integrations, 3) Performance is critical, or 4) You plan to scale rapidly. Many successful startups started with no-code MVPs and rebuilt later. Focus on learning about your customers first, optimize for technical elegance second.",
        "category": "mentorship_needed",
        "text": "### Instruction: Should I build my MVP using no-code tools or custom development?\n### Response: For early validation, no-code tools like Bubble, Webflow, or Airtable can be excellent for MVPs. They allow rapid iteration and cost-effective testing. However, consider custom development if: 1) Your product requires complex algorithms, 2) You need specific integrations, 3) Performance is critical, or 4) You plan to scale rapidly. Many successful startups started with no-code MVPs and rebuilt later. Focus on learning about your customers first, optimize for technical elegance second."
    },
    {
        "instruction": "How do I choose the right technology stack for my startup?",
        "input": "",
        "output": "Choose your tech stack based on: 1) Team expertise - use what your team knows well, 2) Speed to market - prioritize rapid development, 3) Scalability needs - plan for growth but don't over-engineer, 4) Community and support - choose popular, well-documented technologies, 5) Hiring considerations - pick technologies that make it easier to find developers. Popular modern stacks include React/Node.js, Python/Django, or cloud-native solutions. Remember, you can always refactor later - the best technology is the one that helps you validate your business faster.",
        "category": "mentorship_needed",
        "text": "### Instruction: How do I choose the right technology stack for my startup?\n### Response: Choose your tech stack based on: 1) Team expertise - use what your team knows well, 2) Speed to market - prioritize rapid development, 3) Scalability needs - plan for growth but don't over-engineer, 4) Community and support - choose popular, well-documented technologies, 5) Hiring considerations - pick technologies that make it easier to find developers. Popular modern stacks include React/Node.js, Python/Django, or cloud-native solutions. Remember, you can always refactor later - the best technology is the one that helps you validate your business faster."
    }
]

# Sales and marketing questions
sales_marketing_qa = [
    {
        "instruction": "How do I create an effective sales process for my B2B startup?",
        "input": "",
        "output": "Build a systematic B2B sales process: 1) Lead qualification using BANT (Budget, Authority, Need, Timeline), 2) Discovery calls to understand pain points, 3) Demo/presentation tailored to their specific needs, 4) Proposal with clear value proposition and ROI, 5) Negotiation and closing, 6) Onboarding and success tracking. Use a CRM to track every interaction. Start with a longer, consultative sales cycle and optimize for efficiency as you scale. Measure metrics like conversion rates at each stage, average deal size, and sales cycle length.",
        "category": "investment_ready",
        "text": "### Instruction: How do I create an effective sales process for my B2B startup?\n### Response: Build a systematic B2B sales process: 1) Lead qualification using BANT (Budget, Authority, Need, Timeline), 2) Discovery calls to understand pain points, 3) Demo/presentation tailored to their specific needs, 4) Proposal with clear value proposition and ROI, 5) Negotiation and closing, 6) Onboarding and success tracking. Use a CRM to track every interaction. Start with a longer, consultative sales cycle and optimize for efficiency as you scale. Measure metrics like conversion rates at each stage, average deal size, and sales cycle length."
    },
    {
        "instruction": "My marketing campaigns aren't generating quality leads. What should I change?",
        "input": "",
        "output": "Your lead quality issues suggest misalignment between messaging and audience. Review: 1) Target audience definition - are you reaching decision-makers?, 2) Value proposition clarity - does it address specific pain points?, 3) Content quality - is it educational rather than promotional?, 4) Channel selection - are you where your customers spend time?, 5) Lead qualification criteria - are you properly scoring leads?. Consider focusing on one channel first, creating better lead magnets, and implementing lead nurturing sequences. Quality over quantity - better to have 10 qualified leads than 100 unqualified ones.",
        "category": "needs_refinement",
        "text": "### Instruction: My marketing campaigns aren't generating quality leads. What should I change?\n### Response: Your lead quality issues suggest misalignment between messaging and audience. Review: 1) Target audience definition - are you reaching decision-makers?, 2) Value proposition clarity - does it address specific pain points?, 3) Content quality - is it educational rather than promotional?, 4) Channel selection - are you where your customers spend time?, 5) Lead qualification criteria - are you properly scoring leads?. Consider focusing on one channel first, creating better lead magnets, and implementing lead nurturing sequences. Quality over quantity - better to have 10 qualified leads than 100 unqualified ones."
    }
]

# Financial and legal questions
finance_legal_qa = [
    {
        "instruction": "How do I structure equity distribution among co-founders?",
        "input": "",
        "output": "Founder equity should reflect contribution, risk, and future value. Consider: 1) Equal splits (25-25-25-25) for similar contributions, 2) Weighted splits based on role importance, experience, and time commitment, 3) Vesting schedules (typically 4 years with 1-year cliff) to protect against early departures, 4) Reserved equity pool (10-20%) for future employees, 5) Legal documentation through a lawyer. Common mistake: Don't wait too long to formalize. Use tools like the Founder Equity Calculator, but remember that fairness and team harmony are more important than perfect mathematical precision.",
        "category": "mentorship_needed",
        "text": "### Instruction: How do I structure equity distribution among co-founders?\n### Response: Founder equity should reflect contribution, risk, and future value. Consider: 1) Equal splits (25-25-25-25) for similar contributions, 2) Weighted splits based on role importance, experience, and time commitment, 3) Vesting schedules (typically 4 years with 1-year cliff) to protect against early departures, 4) Reserved equity pool (10-20%) for future employees, 5) Legal documentation through a lawyer. Common mistake: Don't wait too long to formalize. Use tools like the Founder Equity Calculator, but remember that fairness and team harmony are more important than perfect mathematical precision."
    },
    {
        "instruction": "What financial metrics should I track at the seed stage?",
        "input": "",
        "output": "At seed stage, focus on: 1) Burn rate and runway - how long can you survive?, 2) Customer acquisition metrics - CAC, conversion rates, organic vs paid growth, 3) Revenue metrics - MRR/ARR for SaaS, transaction volume for marketplaces, 4) Product engagement - DAU/MAU, feature usage, retention cohorts, 5) Unit economics - gross margin, contribution margin per customer. Don't overcomplicate - choose 5-7 key metrics that directly relate to your business model. Track weekly/monthly and focus on trends rather than absolute numbers. These metrics will become crucial for fundraising conversations.",
        "category": "investment_ready",
        "text": "### Instruction: What financial metrics should I track at the seed stage?\n### Response: At seed stage, focus on: 1) Burn rate and runway - how long can you survive?, 2) Customer acquisition metrics - CAC, conversion rates, organic vs paid growth, 3) Revenue metrics - MRR/ARR for SaaS, transaction volume for marketplaces, 4) Product engagement - DAU/MAU, feature usage, retention cohorts, 5) Unit economics - gross margin, contribution margin per customer. Don't overcomplicate - choose 5-7 key metrics that directly relate to your business model. Track weekly/monthly and focus on trends rather than absolute numbers. These metrics will become crucial for fundraising conversations."
    }
]

# Product development questions
product_qa = [
    {
        "instruction": "How do I prioritize features when everything seems important?",
        "input": "",
        "output": "Use a structured prioritization framework: 1) RICE scoring (Reach × Impact × Confidence ÷ Effort), 2) Kano Model to distinguish must-haves from delighters, 3) Customer feedback frequency and urgency, 4) Business impact on key metrics, 5) Strategic alignment with company goals. Start with features that are high impact, low effort, and requested by multiple customers. Remember the 80/20 rule - focus on features that solve core problems for most users. Regular customer interviews and usage analytics should guide decisions more than internal opinions or competitor copying.",
        "category": "mentorship_needed",
        "text": "### Instruction: How do I prioritize features when everything seems important?\n### Response: Use a structured prioritization framework: 1) RICE scoring (Reach × Impact × Confidence ÷ Effort), 2) Kano Model to distinguish must-haves from delighters, 3) Customer feedback frequency and urgency, 4) Business impact on key metrics, 5) Strategic alignment with company goals. Start with features that are high impact, low effort, and requested by multiple customers. Remember the 80/20 rule - focus on features that solve core problems for most users. Regular customer interviews and usage analytics should guide decisions more than internal opinions or competitor copying."
    },
    {
        "instruction": "Our product has low user engagement despite good initial signup rates. How do we fix this?",
        "input": "",
        "output": "Low engagement despite good signups suggests an onboarding or value realization problem. Investigate: 1) Time to value - how quickly do users see benefits?, 2) Onboarding friction - are there too many steps or confusing flows?, 3) Feature discovery - do users understand your core features?, 4) Value proposition alignment - does the product deliver on marketing promises?, 5) User segmentation - different user types may need different approaches. Implement user analytics, run usability tests, and interview both engaged and churned users. Focus on activation metrics and create guided onboarding experiences.",
        "category": "needs_refinement", 
        "text": "### Instruction: Our product has low user engagement despite good initial signup rates. How do we fix this?\n### Response: Low engagement despite good signups suggests an onboarding or value realization problem. Investigate: 1) Time to value - how quickly do users see benefits?, 2) Onboarding friction - are there too many steps or confusing flows?, 3) Feature discovery - do users understand your core features?, 4) Value proposition alignment - does the product deliver on marketing promises?, 5) User segmentation - different user types may need different approaches. Implement user analytics, run usability tests, and interview both engaged and churned users. Focus on activation metrics and create guided onboarding experiences."
    }
]

# Combine all additional conversations
additional_conversations.extend(rwanda_specific)
additional_conversations.extend(technical_qa)
additional_conversations.extend(sales_marketing_qa)
additional_conversations.extend(finance_legal_qa)
additional_conversations.extend(product_qa)

# Add follow-up questions and multi-turn conversations
multi_turn_conversations = [
    {
        "instruction": "I got my first paying customer! What should I do next to build on this success?",
        "input": "",
        "output": "Congratulations! This is a huge milestone. Now focus on: 1) Customer Success - Ensure your first customer gets maximum value and becomes a reference, 2) Case Study - Document their problem, solution, and results for marketing, 3) Feedback Loop - Schedule regular check-ins to gather product feedback, 4) Referrals - Ask for introductions to similar prospects, 5) Optimize Operations - Use this experience to refine your onboarding and delivery processes. Don't immediately focus on scaling - get 5-10 more customers using similar approaches first. Each early customer is a learning opportunity to refine your product-market fit.",
        "category": "mentorship_needed",
        "text": "### Instruction: I got my first paying customer! What should I do next to build on this success?\n### Response: Congratulations! This is a huge milestone. Now focus on: 1) Customer Success - Ensure your first customer gets maximum value and becomes a reference, 2) Case Study - Document their problem, solution, and results for marketing, 3) Feedback Loop - Schedule regular check-ins to gather product feedback, 4) Referrals - Ask for introductions to similar prospects, 5) Optimize Operations - Use this experience to refine your onboarding and delivery processes. Don't immediately focus on scaling - get 5-10 more customers using similar approaches first. Each early customer is a learning opportunity to refine your product-market fit."
    },
    {
        "instruction": "We've hit $100K MRR and want to expand our team. What roles should we hire first?",
        "input": "",
        "output": "At $100K MRR, you're entering a critical growth phase. Prioritize hiring based on your biggest constraints: 1) If sales is the bottleneck, hire a sales manager or BDRs, 2) If product development is slow, add senior engineers, 3) If customer success is suffering, hire a customer success manager, 4) If operations are chaotic, consider an operations manager. Generally, the sequence is: senior engineer → sales hire → customer success → marketing. Ensure each hire can pay for themselves within 6 months through increased revenue or efficiency. Focus on experienced hires who can work independently and eventually manage others.",
        "category": "investment_ready",
        "text": "### Instruction: We've hit $100K MRR and want to expand our team. What roles should we hire first?\n### Response: At $100K MRR, you're entering a critical growth phase. Prioritize hiring based on your biggest constraints: 1) If sales is the bottleneck, hire a sales manager or BDRs, 2) If product development is slow, add senior engineers, 3) If customer success is suffering, hire a customer success manager, 4) If operations are chaotic, consider an operations manager. Generally, the sequence is: senior engineer → sales hire → customer success → marketing. Ensure each hire can pay for themselves within 6 months through increased revenue or efficiency. Focus on experienced hires who can work independently and eventually manage others."
    }
]

additional_conversations.extend(multi_turn_conversations)

# Create more nuanced scenario-based questions
scenario_based = [
    {
        "instruction": "My co-founder wants to quit 6 months into our startup. How should I handle this situation?",
        "input": "",
        "output": "This is a critical situation requiring careful handling: 1) Understand their reasons - is it personal, financial, or disagreement about direction?, 2) Review your founder agreements - what does your vesting schedule and equity documentation say?, 3) Consider alternatives - reduced role, advisory position, or leave of absence, 4) Protect the business - ensure IP assignments, non-compete clauses, and customer relationships are secure, 5) Plan transition - knowledge transfer, team communication, and operational continuity. If they're determined to leave, negotiate fairly on equity clawback and ensure a professional transition that maintains team morale and company reputation.",
        "category": "needs_refinement",
        "text": "### Instruction: My co-founder wants to quit 6 months into our startup. How should I handle this situation?\n### Response: This is a critical situation requiring careful handling: 1) Understand their reasons - is it personal, financial, or disagreement about direction?, 2) Review your founder agreements - what does your vesting schedule and equity documentation say?, 3) Consider alternatives - reduced role, advisory position, or leave of absence, 4) Protect the business - ensure IP assignments, non-compete clauses, and customer relationships are secure, 5) Plan transition - knowledge transfer, team communication, and operational continuity. If they're determined to leave, negotiate fairly on equity clawback and ensure a professional transition that maintains team morale and company reputation."
    },
    {
        "instruction": "A big corporation wants to acquire us after just 12 months. Should we consider it?",
        "input": "",
        "output": "Early acquisition offers require careful evaluation: 1) Valuation assessment - is the offer fair compared to your potential?, 2) Strategic fit - will the acquisition help you achieve your vision faster?, 3) Team considerations - what happens to your employees and culture?, 4) Financial analysis - compare immediate payout vs. long-term potential, 5) Market timing - could you build more value independently?. If the offer is substantial and aligns with your goals, it might make sense. However, if you believe you're building something transformative and the offer doesn't reflect that potential, consider it validation and keep building. Get multiple perspectives from advisors and ensure any decision aligns with your personal and professional goals.",
        "category": "investment_ready",
        "text": "### Instruction: A big corporation wants to acquire us after just 12 months. Should we consider it?\n### Response: Early acquisition offers require careful evaluation: 1) Valuation assessment - is the offer fair compared to your potential?, 2) Strategic fit - will the acquisition help you achieve your vision faster?, 3) Team considerations - what happens to your employees and culture?, 4) Financial analysis - compare immediate payout vs. long-term potential, 5) Market timing - could you build more value independently?. If the offer is substantial and aligns with your goals, it might make sense. However, if you believe you're building something transformative and the offer doesn't reflect that potential, consider it validation and keep building. Get multiple perspectives from advisors and ensure any decision aligns with your personal and professional goals."
    }
]

additional_conversations.extend(scenario_based)

# Combine with existing data
all_conversations = existing_data + additional_conversations

print(f"Original dataset: {len(existing_data)} conversations")
print(f"Additional conversations: {len(additional_conversations)} conversations")
print(f"Total dataset: {len(all_conversations)} conversations")

# Update category distribution
category_counts = {}
for conv in all_conversations:
    category = conv['category']
    category_counts[category] = category_counts.get(category, 0) + 1

print(f"\nFinal category distribution:")
for category, count in category_counts.items():
    print(f"  {category}: {count} examples")

# Save the expanded dataset
with open('data/startup_advisory_dataset_extended.json', 'w') as f:
    json.dump(all_conversations, f, indent=2)

df_extended = pd.DataFrame(all_conversations)
df_extended.to_csv('data/startup_advisory_dataset_extended.csv', index=False)

print(f"\nExtended dataset saved to:")
print("- data/startup_advisory_dataset_extended.json")
print("- data/startup_advisory_dataset_extended.csv")

# Show some examples from each category
print(f"\n=== SAMPLE CONVERSATIONS BY CATEGORY ===")
for category in category_counts.keys():
    category_examples = [conv for conv in all_conversations if conv['category'] == category]
    example = random.choice(category_examples)
    print(f"\n{category.upper()} EXAMPLE:")
    print(f"Q: {example['instruction']}")
    print(f"A: {example['output'][:200]}...")