# First, let's create a comprehensive startup advisory dataset combining multiple sources
import pandas as pd
import json
import numpy as np
from datetime import datetime
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create a robust startup advisory dataset based on Uruti.Rw classification categories
startup_qa_data = []

# Categories aligned with Uruti.Rw: Mentorship Needed, Investment Ready, Needs Refinement
categories = {
    "mentorship_needed": {
        "description": "Questions from entrepreneurs who need guidance and mentorship",
        "questions": [
            "How do I validate my startup idea?",
            "What should I focus on in the early stages?",
            "How do I find my target market?",
            "What are the key metrics I should track?",
            "How do I build a minimum viable product?",
            "What mistakes should I avoid as a first-time entrepreneur?",
            "How do I know if my idea is viable?",
            "What resources do I need to get started?",
            "How do I develop a business model?",
            "What legal considerations should I be aware of?",
            "How do I protect my intellectual property?",
            "What should I include in my business plan?",
            "How do I conduct market research?",
            "What tools can help me organize my startup?",
            "How do I manage my time effectively as a founder?",
            "What skills do I need to develop?",
            "How do I handle uncertainty and risk?",
            "What are the different types of business structures?",
            "How do I price my product or service?",
            "What are the essential elements of a startup?",
            "How do I identify my competition?",
            "What is a value proposition?",
            "How do I create a user persona?",
            "What is the lean startup methodology?",
            "How do I test my assumptions?",
            "What are the stages of startup development?",
            "How do I build credibility for my startup?",
            "What networking strategies should I use?",
            "How do I manage startup stress?",
            "What books should I read as an entrepreneur?"
        ]
    },
    "investment_ready": {
        "description": "Questions from entrepreneurs ready for investment and scaling",
        "questions": [
            "How do I prepare for Series A funding?",
            "What metrics do investors look for?",
            "How do I create a compelling pitch deck?",
            "What valuation should I expect?",
            "How do I negotiate with investors?",
            "What are the different types of funding?",
            "How do I find angel investors?",
            "What due diligence should I expect?",
            "How do I scale my team effectively?",
            "What are the key performance indicators for growth?",
            "How do I expand to new markets?",
            "What operational systems should I implement?",
            "How do I maintain company culture during growth?",
            "What financial controls do I need?",
            "How do I manage cash flow during scaling?",
            "What partnerships should I consider?",
            "How do I optimize my sales funnel?",
            "What technology infrastructure do I need?",
            "How do I handle international expansion?",
            "What exit strategies should I consider?",
            "How do I prepare for an IPO?",
            "What are the risks of rapid scaling?",
            "How do I maintain product quality during growth?",
            "What customer retention strategies work best?",
            "How do I build strategic partnerships?",
            "What acquisitions should I consider?",
            "How do I manage investor relations?",
            "What governance structures do I need?",
            "How do I handle market competition at scale?",
            "What are the signs of product-market fit?"
        ]
    },
    "needs_refinement": {
        "description": "Questions from entrepreneurs whose ideas need improvement",
        "questions": [
            "How do I pivot my business model?",
            "What should I do if my product isn't selling?",
            "How do I improve customer retention?",
            "What if my target market is too small?",
            "How do I fix poor unit economics?",
            "What should I do about negative customer feedback?",
            "How do I revise my value proposition?",
            "What if my competition is outperforming me?",
            "How do I improve my product features?",
            "What should I do if I'm running out of funding?",
            "How do I reduce customer acquisition costs?",
            "What if my team isn't performing well?",
            "How do I improve my marketing strategy?",
            "What should I do about technical debt?",
            "How do I handle a failing partnership?",
            "What if my business model isn't sustainable?",
            "How do I improve operational efficiency?",
            "What should I do about low conversion rates?",
            "How do I fix pricing strategy issues?",
            "What if my market positioning is wrong?",
            "How do I improve customer satisfaction?",
            "What should I do about high churn rates?",
            "How do I optimize my supply chain?",
            "What if my product roadmap is unclear?",
            "How do I improve team communication?",
            "What should I do about regulatory challenges?",
            "How do I fix brand reputation issues?",
            "What if my growth has stagnated?",
            "How do I improve cash flow management?",
            "What should I do about market timing issues?"
        ]
    }
}

# Generate comprehensive responses for each category
responses = {
    "mentorship_needed": [
        "As a mentor, I'd recommend starting with customer discovery. Talk to at least 50 potential customers to validate your problem hypothesis. Document their pain points and willingness to pay for a solution.",
        "Focus on building an MVP (Minimum Viable Product) first. This should be the simplest version that solves the core problem. Avoid feature bloat in the early stages.",
        "Conduct primary research through surveys, interviews, and observational studies. Use tools like Google Analytics, social media insights, and competitor analysis to understand your market.",
        "Track key metrics like customer acquisition cost (CAC), lifetime value (LTV), monthly recurring revenue (MRR), and churn rate. These will guide your decision-making.",
        "Start with a simple solution that addresses one core problem well. Use no-code tools, prototypes, or even manual processes to validate demand before building.",
        "Avoid perfectionism, building without customer input, ignoring cash flow, hiring too quickly, and neglecting legal foundations. Focus on learning and iteration.",
        "Test your assumptions through experiments. Look for strong customer demand, clear monetization path, scalable business model, and your ability to execute effectively.",
        "You'll need domain expertise, basic financial understanding, network of advisors, minimal viable team, and sufficient runway to reach key milestones.",
        "Choose between B2B, B2C, subscription, marketplace, or freemium models based on your customer type and value delivery method. Test and iterate.",
        "Register your business entity, understand tax obligations, protect intellectual property, draft founder agreements, and comply with industry regulations.",
        "File for patents or trademarks if applicable, use non-disclosure agreements, document your IP creation process, and consider trade secret protection strategies.",
        "Include executive summary, market analysis, competitive landscape, business model, financial projections, marketing strategy, team overview, and funding requirements.",
        "Use both primary (surveys, interviews) and secondary research (industry reports, competitor analysis). Focus on market size, customer needs, and competitive positioning.",
        "Use project management tools like Notion or Asana, financial tracking with QuickBooks, customer relationship management systems, and communication platforms like Slack.",
        "Prioritize tasks using frameworks like RICE or MoSCoW, time-block your calendar, delegate effectively, and maintain work-life balance to prevent burnout.",
        "Develop leadership, sales, financial literacy, product management, and communication skills. Consider taking courses or finding mentors in these areas.",
        "Build financial reserves, diversify revenue streams, create contingency plans, maintain strong relationships, and focus on learning from failures rather than avoiding them.",
        "Choose between sole proprietorship, LLC, corporation, or partnership based on liability protection, tax implications, growth plans, and investor requirements.",
        "Research competitor pricing, calculate your costs plus margin, test different price points, consider value-based pricing, and be prepared to iterate based on market feedback.",
        "Essential elements include a clear value proposition, viable business model, target market understanding, competitive advantage, founding team, and execution capability."
    ],
    "investment_ready": [
        "Prepare detailed financial projections, demonstrate strong unit economics, show consistent growth metrics, build a experienced team, and create a compelling growth story for investors.",
        "Investors focus on market size, revenue growth, customer retention, unit economics, team experience, competitive moats, and scalability potential. Ensure your metrics tell a compelling story.",
        "Create a 10-12 slide deck covering problem, solution, market opportunity, business model, traction, competition, team, financials, funding ask, and use of funds.",
        "Valuation depends on stage, traction, market size, and comparables. Early-stage companies typically trade at 10-20x revenue multiples, but this varies significantly by industry.",
        "Understand your leverage, prepare multiple term sheet options, focus on more than just valuation, consider board composition and liquidation preferences carefully.",
        "Options include bootstrapping, angel investors, venture capital, crowdfunding, debt financing, grants, and strategic partnerships. Choose based on your stage and needs.",
        "Network through industry events, accelerators, online platforms like AngelList, warm introductions from other entrepreneurs, and angel investor groups in your area.",
        "Expect review of financials, legal documents, customer references, technical architecture, IP portfolio, team backgrounds, and market analysis. Prepare comprehensive documentation.",
        "Hire for culture fit and growth potential, implement scalable HR processes, create clear role definitions, establish performance management systems, and maintain team cohesion.",
        "Focus on North Star metrics like revenue, user growth, retention, NPS scores, market share, operational efficiency, and team performance indicators that drive long-term value.",
        "Conduct thorough market research, adapt your product for local needs, establish local partnerships, understand regulatory requirements, and build region-specific marketing strategies.",
        "Implement ERP systems, standardize processes, create detailed documentation, establish quality controls, and build scalable customer support and fulfillment operations.",
        "Document your values clearly, hire culture champions, create rituals and traditions, maintain transparent communication, and ensure leadership models desired behaviors consistently.",
        "Implement financial planning software, establish approval workflows, create detailed budgets and forecasts, maintain strong audit trails, and separate duties appropriately.",
        "Create detailed cash flow forecasts, maintain sufficient reserves, optimize payment terms, implement efficient collections, and establish credit facilities before you need them.",
        "Consider strategic partnerships for distribution, technology integrations, co-marketing opportunities, or complementary product offerings that enhance your value proposition.",
        "Analyze each funnel stage, A/B test improvements, implement marketing automation, optimize conversion points, and focus on reducing friction in the customer journey.",
        "Build cloud-native architecture, implement robust security, create scalable databases, establish DevOps practices, and ensure high availability and disaster recovery capabilities.",
        "Develop local market entry strategies, establish legal entities, understand cultural differences, build local teams, and adapt your product for international markets.",
        "Consider IPO, acquisition, or management buyout based on your goals, market conditions, and business performance. Prepare years in advance for any exit strategy."
    ],
    "needs_refinement": [
        "Analyze what's not working, gather customer feedback, explore adjacent markets, test new business models, and be prepared to make significant changes to find product-market fit.",
        "Conduct customer interviews to understand why they're not buying, analyze your value proposition, review your pricing strategy, improve your marketing message, and consider product improvements.",
        "Analyze churn reasons, improve onboarding, create customer success programs, gather regular feedback, build loyalty programs, and focus on delivering consistent value.",
        "Consider expanding your target market, finding niche applications, exploring B2B opportunities, or pivoting to adjacent markets with larger addressable audiences.",
        "Analyze your cost structure, improve pricing strategy, optimize customer acquisition channels, reduce churn, increase customer lifetime value, and eliminate unprofitable segments.",
        "Listen carefully to feedback, identify patterns, prioritize critical issues, communicate your improvement plans, and show customers you value their input through action.",
        "Conduct fresh customer interviews, analyze successful competitors, test new messaging, focus on unique benefits, and ensure your value proposition addresses real customer pain points.",
        "Analyze their advantages, identify your unique strengths, improve your product differentiation, enhance customer experience, optimize your marketing, or find underserved niches.",
        "Prioritize features based on customer feedback, conduct usability testing, implement agile development processes, gather continuous feedback, and focus on core functionality first.",
        "Extend your runway through cost reduction, explore revenue-based financing, pitch to new investors, consider strategic partnerships, or pursue grants and competitions.",
        "Optimize your marketing channels, improve conversion rates, enhance referral programs, focus on organic growth strategies, and analyze customer lifetime value to justify acquisition costs.",
        "Assess team members individually, provide additional training or support, clarify roles and expectations, consider team restructuring, and focus on cultural fit and performance.",
        "Analyze your current strategy's performance, test new channels, improve your messaging, enhance your brand positioning, and focus on channels with the best ROI.",
        "Prioritize critical issues, allocate time for refactoring, implement better development practices, consider bringing in technical expertise, and balance new features with improvements.",
        "Evaluate the partnership's strategic value, communicate issues openly, renegotiate terms if needed, or consider ending partnerships that aren't delivering mutual value.",
        "Review your unit economics, explore new revenue streams, reduce operational costs, improve customer lifetime value, and consider fundamental business model changes.",
        "Identify bottlenecks, automate repetitive processes, implement better tools and systems, optimize workflows, and continuously measure and improve key operational metrics.",
        "Analyze your funnel, A/B test different approaches, improve your value proposition, optimize your website and marketing materials, and remove friction from the conversion process.",
        "Analyze competitor pricing, survey your customers about price sensitivity, test different pricing models, consider value-based pricing, and ensure your pricing reflects your value.",
        "Conduct new market research, test different positioning messages, analyze successful competitors, and consider repositioning around different customer segments or value propositions."
    ]
}

# Create the comprehensive dataset
for category, data in categories.items():
    questions = data["questions"]
    category_responses = responses[category]
    
    for i, question in enumerate(questions):
        response = category_responses[i % len(category_responses)]
        
        startup_qa_data.append({
            "instruction": question,
            "input": "",
            "output": response,
            "category": category,
            "text": f"### Instruction: {question}\n### Response: {response}"
        })

# Add additional variations and more complex scenarios
complex_scenarios = [
    {
        "instruction": "I'm a first-time founder with a SaaS idea for small businesses. I've built a prototype but haven't gotten any paying customers yet. What should I focus on?",
        "input": "",
        "output": "Based on your stage, you're in the 'Mentorship Needed' category. Focus on customer discovery first - reach out to 50+ small business owners to validate your problem hypothesis. Don't just ask if they like your solution; ask about their current workflows and pain points. Once you understand their needs better, refine your prototype to address the most critical issues. Then, offer a free trial or pilot program to gather real usage data and feedback. Remember, at this stage, learning is more valuable than revenue.",
        "category": "mentorship_needed",
        "text": "### Instruction: I'm a first-time founder with a SaaS idea for small businesses. I've built a prototype but haven't gotten any paying customers yet. What should I focus on?\n### Response: Based on your stage, you're in the 'Mentorship Needed' category. Focus on customer discovery first - reach out to 50+ small business owners to validate your problem hypothesis. Don't just ask if they like your solution; ask about their current workflows and pain points. Once you understand their needs better, refine your prototype to address the most critical issues. Then, offer a free trial or pilot program to gather real usage data and feedback. Remember, at this stage, learning is more valuable than revenue."
    },
    {
        "instruction": "My startup has $2M ARR, 40% YoY growth, and strong unit economics. I want to raise Series A to scale our sales team and expand internationally. How should I prepare?",
        "input": "",
        "output": "Congratulations! You're clearly 'Investment Ready'. For Series A preparation, focus on: 1) Financial metrics: Ensure your LTV/CAC ratio is >3, gross margin >70%, and net revenue retention >100%. 2) Create a compelling pitch deck highlighting your TAM expansion, competitive moats, and international expansion strategy. 3) Build relationships with tier-1 VCs 6 months before fundraising. 4) Prepare detailed financial projections showing path to $10M+ ARR. 5) Document your go-to-market playbook to demonstrate repeatability. Target $5-15M raise based on your metrics and market opportunity.",
        "category": "investment_ready", 
        "text": "### Instruction: My startup has $2M ARR, 40% YoY growth, and strong unit economics. I want to raise Series A to scale our sales team and expand internationally. How should I prepare?\n### Response: Congratulations! You're clearly 'Investment Ready'. For Series A preparation, focus on: 1) Financial metrics: Ensure your LTV/CAC ratio is >3, gross margin >70%, and net revenue retention >100%. 2) Create a compelling pitch deck highlighting your TAM expansion, competitive moats, and international expansion strategy. 3) Build relationships with tier-1 VCs 6 months before fundraising. 4) Prepare detailed financial projections showing path to $10M+ ARR. 5) Document your go-to-market playbook to demonstrate repeatability. Target $5-15M raise based on your metrics and market opportunity."
    },
    {
        "instruction": "Our startup launched 18 months ago but we're struggling with high churn rates (15% monthly), low conversion from trial to paid (2%), and our CAC is higher than our LTV. What should we do?",
        "input": "",
        "output": "Your startup falls into the 'Needs Refinement' category and requires immediate attention to unit economics. Here's a systematic approach: 1) Churn Analysis: Interview churned customers to identify patterns - is it product fit, onboarding, or value realization? 2) Conversion Optimization: Analyze your trial-to-paid funnel, improve onboarding experience, and add more value during trial period. 3) Cost Reduction: Pause expensive acquisition channels, focus on organic growth and referrals. 4) Product-Market Fit: You may need to pivot your target market or core features. 5) Consider a strategic pause on growth to fix fundamentals. These metrics suggest fundamental issues that need addressing before scaling further.",
        "category": "needs_refinement",
        "text": "### Instruction: Our startup launched 18 months ago but we're struggling with high churn rates (15% monthly), low conversion from trial to paid (2%), and our CAC is higher than our LTV. What should we do?\n### Response: Your startup falls into the 'Needs Refinement' category and requires immediate attention to unit economics. Here's a systematic approach: 1) Churn Analysis: Interview churned customers to identify patterns - is it product fit, onboarding, or value realization? 2) Conversion Optimization: Analyze your trial-to-paid funnel, improve onboarding experience, and add more value during trial period. 3) Cost Reduction: Pause expensive acquisition channels, focus on organic growth and referrals. 4) Product-Market Fit: You may need to pivot your target market or core features. 5) Consider a strategic pause on growth to fix fundamentals. These metrics suggest fundamental issues that need addressing before scaling further."
    }
]

startup_qa_data.extend(complex_scenarios)

print(f"Generated {len(startup_qa_data)} startup advisory Q&A pairs")
print(f"Categories distribution:")
for category in categories.keys():
    count = len([item for item in startup_qa_data if item['category'] == category])
    print(f"  {category}: {count} examples")

# Save the dataset
import os
os.makedirs('data', exist_ok=True)
with open('data/startup_advisory_dataset.json', 'w') as f:
    json.dump(startup_qa_data, f, indent=2)

# Create a CSV version as well for easy viewing
df = pd.DataFrame(startup_qa_data)
df.to_csv('data/startup_advisory_dataset.csv', index=False)

print("\nDataset saved to:")
print("- data/startup_advisory_dataset.json")  
print("- data/startup_advisory_dataset.csv")
print(f"\nFirst 3 examples:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Category: {startup_qa_data[i]['category']}")
    print(f"Question: {startup_qa_data[i]['instruction']}")
    print(f"Answer: {startup_qa_data[i]['output'][:200]}...")