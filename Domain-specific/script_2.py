# Let's create an even more comprehensive dataset by adding more realistic conversational patterns
# and integrating startup interview data patterns

import json
import pandas as pd
import random
from datetime import datetime

# Load our existing extended dataset
with open('data/startup_advisory_dataset_extended.json', 'r') as f:
    base_data = json.load(f)

# Add Y Combinator style startup advice patterns
yc_style_qa = [
    {
        "instruction": "How do I know if I should pivot or persevere?",
        "input": "",
        "output": "The pivot vs persevere decision depends on key signals: 1) Customer engagement - are users actually using your product regularly?, 2) Growth rate - is organic growth accelerating or flattering?, 3) Market feedback - are customers asking for fundamental changes or just features?, 4) Founder conviction - do you still believe in the problem?, 5) Resource runway - how much time do you have to experiment?. Pivot if you're not seeing organic growth after 6+ months of iteration, customers don't seem to care when you go away, or you've lost conviction in the problem. Persevere if you're seeing slow but consistent growth, strong user engagement, and clear path to product-market fit.",
        "category": "needs_refinement",
        "text": "### Instruction: How do I know if I should pivot or persevere?\n### Response: The pivot vs persevere decision depends on key signals: 1) Customer engagement - are users actually using your product regularly?, 2) Growth rate - is organic growth accelerating or flattering?, 3) Market feedback - are customers asking for fundamental changes or just features?, 4) Founder conviction - do you still believe in the problem?, 5) Resource runway - how much time do you have to experiment?. Pivot if you're not seeing organic growth after 6+ months of iteration, customers don't seem to care when you go away, or you've lost conviction in the problem. Persevere if you're seeing slow but consistent growth, strong user engagement, and clear path to product-market fit."
    },
    {
        "instruction": "What's the biggest mistake first-time founders make?",
        "input": "",
        "output": "The biggest mistake is building a product without talking to customers first. Many founders fall in love with their solution and spend months building features nobody wants. Other critical mistakes include: 1) Perfectionism - launching too late with too many features, 2) Wrong co-founder choice - not aligning on vision and work ethic, 3) Ignoring unit economics - focusing on growth without understanding profitability, 4) Premature scaling - hiring and spending before finding product-market fit, 5) Not talking to users enough - making decisions based on assumptions rather than data. The antidote is customer obsession from day one.",
        "category": "mentorship_needed",
        "text": "### Instruction: What's the biggest mistake first-time founders make?\n### Response: The biggest mistake is building a product without talking to customers first. Many founders fall in love with their solution and spend months building features nobody wants. Other critical mistakes include: 1) Perfectionism - launching too late with too many features, 2) Wrong co-founder choice - not aligning on vision and work ethic, 3) Ignoring unit economics - focusing on growth without understanding profitability, 4) Premature scaling - hiring and spending before finding product-market fit, 5) Not talking to users enough - making decisions based on assumptions rather than data. The antidote is customer obsession from day one."
    },
    {
        "instruction": "How do I build a great product culture in my startup?",
        "input": "",
        "output": "Great product culture starts with customer obsession and data-driven decision making: 1) Make everyone customer-facing - engineers should talk to users regularly, 2) Ship fast and measure - prefer quick iterations over perfect launches, 3) Data transparency - make key metrics visible to everyone, 4) Customer feedback loops - built regular user research into your process, 5) Cross-functional collaboration - break down silos between teams, 6) Embrace failure - celebrate learning from experiments that don't work. The CEO should model this behavior and hire people who naturally think about customer impact. Product decisions should always start with 'how does this help our users?'",
        "category": "investment_ready",
        "text": "### Instruction: How do I build a great product culture in my startup?\n### Response: Great product culture starts with customer obsession and data-driven decision making: 1) Make everyone customer-facing - engineers should talk to users regularly, 2) Ship fast and measure - prefer quick iterations over perfect launches, 3) Data transparency - make key metrics visible to everyone, 4) Customer feedback loops - built regular user research into your process, 5) Cross-functional collaboration - break down silos between teams, 6) Embrace failure - celebrate learning from experiments that don't work. The CEO should model this behavior and hire people who naturally think about customer impact. Product decisions should always start with 'how does this help our users?'"
    }
]

# Add investor perspective questions
investor_perspective_qa = [
    {
        "instruction": "What do VCs really look for in early-stage startups?",
        "input": "",
        "output": "VCs evaluate startups across several dimensions: 1) Market size - is this a $1B+ opportunity?, 2) Team quality - can these founders execute and scale?, 3) Product traction - evidence of product-market fit and growth, 4) Business model - clear path to significant revenue, 5) Competitive advantage - sustainable moats, 6) Timing - why now?, 7) Vision - can this become a category-defining company?. Early stage VCs weight team heavily - they bet on founders more than products. Show strong unit economics, consistent growth, and deep customer understanding. Most importantly, demonstrate you can build something customers love and will pay for.",
        "category": "investment_ready",
        "text": "### Instruction: What do VCs really look for in early-stage startups?\n### Response: VCs evaluate startups across several dimensions: 1) Market size - is this a $1B+ opportunity?, 2) Team quality - can these founders execute and scale?, 3) Product traction - evidence of product-market fit and growth, 4) Business model - clear path to significant revenue, 5) Competitive advantage - sustainable moats, 6) Timing - why now?, 7) Vision - can this become a category-defining company?. Early stage VCs weight team heavily - they bet on founders more than products. Show strong unit economics, consistent growth, and deep customer understanding. Most importantly, demonstrate you can build something customers love and will pay for."
    },
    {
        "instruction": "How do I create a compelling pitch deck that stands out?",
        "input": "",
        "output": "A compelling pitch deck tells a story in 10-12 slides: 1) Problem - paint a vivid picture of customer pain, 2) Solution - show your unique approach, 3) Market - size and growth potential, 4) Product - demo the core value, 5) Traction - prove customers want this, 6) Business Model - how you make money, 7) Competition - why you'll win, 8) Team - why you're the right people, 9) Financials - realistic projections, 10) Funding ask - specific use of funds. Keep it visual, tell stories not statistics, practice obsessively, and always end with a clear ask. The goal is to get a meeting, not close the deal in the deck.",
        "category": "investment_ready",
        "text": "### Instruction: How do I create a compelling pitch deck that stands out?\n### Response: A compelling pitch deck tells a story in 10-12 slides: 1) Problem - paint a vivid picture of customer pain, 2) Solution - show your unique approach, 3) Market - size and growth potential, 4) Product - demo the core value, 5) Traction - prove customers want this, 6) Business Model - how you make money, 7) Competition - why you'll win, 8) Team - why you're the right people, 9) Financials - realistic projections, 10) Funding ask - specific use of funds. Keep it visual, tell stories not statistics, practice obsessively, and always end with a clear ask. The goal is to get a meeting, not close the deal in the deck."
    }
]

# Add operational and scaling questions
operations_scaling_qa = [
    {
        "instruction": "How do I maintain quality while scaling quickly?",
        "input": "",
        "output": "Scaling while maintaining quality requires systematic approaches: 1) Document everything - create playbooks and standard operating procedures, 2) Implement quality controls - automated testing, peer reviews, customer feedback loops, 3) Hire for culture fit - people who naturally maintain high standards, 4) Create feedback systems - regular customer surveys, internal quality metrics, 5) Invest in tools and infrastructure - don't let technology be the bottleneck, 6) Scale incrementally - test processes with small groups before full rollout. The key is building quality into your processes rather than inspecting it in later. Plan for scale early but don't over-engineer before you need it.",
        "category": "investment_ready",
        "text": "### Instruction: How do I maintain quality while scaling quickly?\n### Response: Scaling while maintaining quality requires systematic approaches: 1) Document everything - create playbooks and standard operating procedures, 2) Implement quality controls - automated testing, peer reviews, customer feedback loops, 3) Hire for culture fit - people who naturally maintain high standards, 4) Create feedback systems - regular customer surveys, internal quality metrics, 5) Invest in tools and infrastructure - don't let technology be the bottleneck, 6) Scale incrementally - test processes with small groups before full rollout. The key is building quality into your processes rather than inspecting it in later. Plan for scale early but don't over-engineer before you need it."
    },
    {
        "instruction": "My startup is growing fast but profitability is declining. What's wrong?",
        "input": "",
        "output": "Declining profitability during growth often indicates unit economics problems: 1) Customer acquisition costs rising faster than lifetime value, 2) Increased operational complexity creating inefficiencies, 3) Price competition forcing margin compression, 4) Infrastructure costs scaling non-linearly, 5) Quality issues leading to higher support costs. Immediate actions: audit your unit economics by cohort, identify which growth channels are profitable, optimize your highest-volume processes, and consider raising prices or reducing costs. Sometimes slowing growth to fix fundamentals is better than scaling unprofitably. Focus on sustainable growth over vanity metrics.",
        "category": "needs_refinement",
        "text": "### Instruction: My startup is growing fast but profitability is declining. What's wrong?\n### Response: Declining profitability during growth often indicates unit economics problems: 1) Customer acquisition costs rising faster than lifetime value, 2) Increased operational complexity creating inefficiencies, 3) Price competition forcing margin compression, 4) Infrastructure costs scaling non-linearly, 5) Quality issues leading to higher support costs. Immediate actions: audit your unit economics by cohort, identify which growth channels are profitable, optimize your highest-volume processes, and consider raising prices or reducing costs. Sometimes slowing growth to fix fundamentals is better than scaling unprofitably. Focus on sustainable growth over vanity metrics."
    }
]

# Add crisis management and difficult situations
crisis_management_qa = [
    {
        "instruction": "A key team member just left during a critical project. How do I handle this?",
        "input": "",
        "output": "Losing a key team member during a critical project requires immediate action: 1) Assess impact - what knowledge, relationships, and work will be lost?, 2) Secure assets - ensure all work, passwords, and IP are transferred, 3) Communicate transparently - tell the team and stakeholders honestly, 4) Redistribute work - identify who can cover critical functions temporarily, 5) Accelerate hiring - start recruiting immediately for permanent replacement, 6) Learn from exit - understand why they left to prevent future departures. Consider hiring contractors or consultants for immediate gaps. Use this as an opportunity to document processes and reduce single points of failure.",
        "category": "needs_refinement",
        "text": "### Instruction: A key team member just left during a critical project. How do I handle this?\n### Response: Losing a key team member during a critical project requires immediate action: 1) Assess impact - what knowledge, relationships, and work will be lost?, 2) Secure assets - ensure all work, passwords, and IP are transferred, 3) Communicate transparently - tell the team and stakeholders honestly, 4) Redistribute work - identify who can cover critical functions temporarily, 5) Accelerate hiring - start recruiting immediately for permanent replacement, 6) Learn from exit - understand why they left to prevent future departures. Consider hiring contractors or consultants for immediate gaps. Use this as an opportunity to document processes and reduce single points of failure."
    },
    {
        "instruction": "We just discovered a major security breach in our product. What should we do?",
        "input": "",
        "output": "A security breach requires immediate, systematic response: 1) Contain the breach - stop ongoing access and secure systems, 2) Assess impact - understand what data was accessed and which users affected, 3) Legal compliance - notify authorities as required (GDPR, etc.), 4) Customer communication - inform affected users transparently and quickly, 5) Fix vulnerabilities - patch the security holes and improve systems, 6) External help - engage security experts and legal counsel. Be transparent about what happened, what you're doing to fix it, and how you'll prevent future breaches. Your response will define customer trust going forward. Consider this a learning opportunity to build better security practices.",
        "category": "needs_refinement",
        "text": "### Instruction: We just discovered a major security breach in our product. What should we do?\n### Response: A security breach requires immediate, systematic response: 1) Contain the breach - stop ongoing access and secure systems, 2) Assess impact - understand what data was accessed and which users affected, 3) Legal compliance - notify authorities as required (GDPR, etc.), 4) Customer communication - inform affected users transparently and quickly, 5) Fix vulnerabilities - patch the security holes and improve systems, 6) External help - engage security experts and legal counsel. Be transparent about what happened, what you're doing to fix it, and how you'll prevent future breaches. Your response will define customer trust going forward. Consider this a learning opportunity to build better security practices."
    }
]

# Add domain-specific questions for different industries
industry_specific_qa = [
    {
        "instruction": "How do I navigate regulatory requirements for a fintech startup?",
        "input": "",
        "output": "Fintech regulatory compliance is complex but manageable with the right approach: 1) Identify applicable regulations - banking, securities, anti-money laundering, data protection, 2) Engage legal experts early - don't try to navigate this alone, 3) Build compliance into your product - design systems to meet regulatory requirements from day one, 4) Establish relationships with regulators - participate in regulatory sandboxes if available, 5) Plan for licensing - understand timeline and costs for required licenses, 6) Consider partnerships - work with licensed institutions to reduce regulatory burden. Start with legal research and expert consultation before building. Compliance costs should be factored into your financial planning and fundraising.",
        "category": "mentorship_needed",
        "text": "### Instruction: How do I navigate regulatory requirements for a fintech startup?\n### Response: Fintech regulatory compliance is complex but manageable with the right approach: 1) Identify applicable regulations - banking, securities, anti-money laundering, data protection, 2) Engage legal experts early - don't try to navigate this alone, 3) Build compliance into your product - design systems to meet regulatory requirements from day one, 4) Establish relationships with regulators - participate in regulatory sandboxes if available, 5) Plan for licensing - understand timeline and costs for required licenses, 6) Consider partnerships - work with licensed institutions to reduce regulatory burden. Start with legal research and expert consultation before building. Compliance costs should be factored into your financial planning and fundraising."
    },
    {
        "instruction": "What's unique about building a B2B SaaS startup versus other business models?",
        "input": "",
        "output": "B2B SaaS has distinct characteristics that affect strategy: 1) Longer sales cycles - expect 3-18 months from first contact to close, 2) Higher customer lifetime values - justify more expensive acquisition approaches, 3) Recurring revenue - focus on retention and expansion over new customer acquisition, 4) Product stickiness - switching costs create competitive advantages, 5) Enterprise needs - security, integrations, and customization become critical. Key metrics differ too - focus on MRR, churn, expansion revenue, and customer health scores rather than just user growth. The path to product-market fit involves deeper customer relationships and more complex implementation cycles, but the reward is predictable, scalable revenue.",
        "category": "mentorship_needed", 
        "text": "### Instruction: What's unique about building a B2B SaaS startup versus other business models?\n### Response: B2B SaaS has distinct characteristics that affect strategy: 1) Longer sales cycles - expect 3-18 months from first contact to close, 2) Higher customer lifetime values - justify more expensive acquisition approaches, 3) Recurring revenue - focus on retention and expansion over new customer acquisition, 4) Product stickiness - switching costs create competitive advantages, 5) Enterprise needs - security, integrations, and customization become critical. Key metrics differ too - focus on MRR, churn, expansion revenue, and customer health scores rather than just user growth. The path to product-market fit involves deeper customer relationships and more complex implementation cycles, but the reward is predictable, scalable revenue."
    }
]

# Combine all new conversations
new_conversations = []
new_conversations.extend(yc_style_qa)
new_conversations.extend(investor_perspective_qa)
new_conversations.extend(operations_scaling_qa)
new_conversations.extend(crisis_management_qa)
new_conversations.extend(industry_specific_qa)

# Add more conversational and informal questions
informal_qa = [
    {
        "instruction": "I'm feeling burned out as a founder. How do I deal with this?",
        "input": "",
        "output": "Founder burnout is incredibly common and needs immediate attention: 1) Acknowledge it's real - burnout isn't a weakness, it's a signal, 2) Take breaks - even short ones can help reset your perspective, 3) Delegate more - identify tasks only you can do vs. tasks others can handle, 4) Build support systems - join founder groups, find mentors, or get professional help, 5) Reassess priorities - what really needs your attention right now?, 6) Exercise and sleep - physical health directly affects mental resilience. Remember, taking care of yourself isn't selfish - your startup needs you at your best. Consider it an investment in your company's success. If burnout persists, taking a longer break might be necessary.",
        "category": "mentorship_needed",
        "text": "### Instruction: I'm feeling burned out as a founder. How do I deal with this?\n### Response: Founder burnout is incredibly common and needs immediate attention: 1) Acknowledge it's real - burnout isn't a weakness, it's a signal, 2) Take breaks - even short ones can help reset your perspective, 3) Delegate more - identify tasks only you can do vs. tasks others can handle, 4) Build support systems - join founder groups, find mentors, or get professional help, 5) Reassess priorities - what really needs your attention right now?, 6) Exercise and sleep - physical health directly affects mental resilience. Remember, taking care of yourself isn't selfish - your startup needs you at your best. Consider it an investment in your company's success. If burnout persists, taking a longer break might be necessary."
    },
    {
        "instruction": "Should I quit my day job to focus on my startup full-time?",
        "input": "",
        "output": "The timing for going full-time depends on several factors: 1) Financial runway - do you have 6-12 months of expenses saved?, 2) Business traction - are you seeing consistent customer demand?, 3) Time constraints - is your day job preventing meaningful progress?, 4) Risk tolerance - can you handle the financial and emotional uncertainty?, 5) Family situation - do you have dependents or major financial obligations?. Consider a gradual transition - reduce to part-time, take unpaid leave, or negotiate flexible arrangements first. The goal is having enough time to make real progress while maintaining financial stability. There's no shame in keeping your day job longer if needed - better to build sustainably than risk everything prematurely.",
        "category": "mentorship_needed",
        "text": "### Instruction: Should I quit my day job to focus on my startup full-time?\n### Response: The timing for going full-time depends on several factors: 1) Financial runway - do you have 6-12 months of expenses saved?, 2) Business traction - are you seeing consistent customer demand?, 3) Time constraints - is your day job preventing meaningful progress?, 4) Risk tolerance - can you handle the financial and emotional uncertainty?, 5) Family situation - do you have dependents or major financial obligations?. Consider a gradual transition - reduce to part-time, take unpaid leave, or negotiate flexible arrangements first. The goal is having enough time to make real progress while maintaining financial stability. There's no shame in keeping your day job longer if needed - better to build sustainably than risk everything prematurely."
    }
]

new_conversations.extend(informal_qa)

# Create final comprehensive dataset
final_dataset = base_data + new_conversations

print(f"Base dataset: {len(base_data)} conversations")
print(f"New conversations added: {len(new_conversations)} conversations") 
print(f"Final comprehensive dataset: {len(final_dataset)} conversations")

# Update category distribution
final_category_counts = {}
for conv in final_dataset:
    category = conv['category']
    final_category_counts[category] = final_category_counts.get(category, 0) + 1

print(f"\nFinal category distribution:")
for category, count in final_category_counts.items():
    print(f"  {category}: {count} examples ({count/len(final_dataset)*100:.1f}%)")

# Save the comprehensive dataset
with open('data/startup_advisory_comprehensive_dataset.json', 'w') as f:
    json.dump(final_dataset, f, indent=2)

df_final = pd.DataFrame(final_dataset)
df_final.to_csv('data/startup_advisory_comprehensive_dataset.csv', index=False)

# Create train/validation/test splits
random.shuffle(final_dataset)
train_size = int(0.8 * len(final_dataset))
val_size = int(0.1 * len(final_dataset))

train_data = final_dataset[:train_size]
val_data = final_dataset[train_size:train_size + val_size]
test_data = final_dataset[train_size + val_size:]

# Save splits
with open('data/train_data.json', 'w') as f:
    json.dump(train_data, f, indent=2)
    
with open('data/val_data.json', 'w') as f:
    json.dump(val_data, f, indent=2)
    
with open('data/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"\nDataset splits:")
print(f"  Training: {len(train_data)} examples ({len(train_data)/len(final_dataset)*100:.1f}%)")
print(f"  Validation: {len(val_data)} examples ({len(val_data)/len(final_dataset)*100:.1f}%)")
print(f"  Test: {len(test_data)} examples ({len(test_data)/len(final_dataset)*100:.1f}%)")

print(f"\nFiles created:")
print("- data/startup_advisory_comprehensive_dataset.json")
print("- data/startup_advisory_comprehensive_dataset.csv")  
print("- data/train_data.json")
print("- data/val_data.json")
print("- data/test_data.json")

# Display sample from each category
print(f"\n=== FINAL SAMPLE CONVERSATIONS ===")
for category in final_category_counts.keys():
    category_examples = [conv for conv in final_dataset if conv['category'] == category]
    example = random.choice(category_examples)
    print(f"\n{category.upper().replace('_', ' ')} EXAMPLE:")
    print(f"Q: {example['instruction']}")
    print(f"A: {example['output'][:150]}...")

print(f"\n📊 DATASET STATISTICS:")
print(f"Total conversations: {len(final_dataset)}")
print(f"Average response length: {np.mean([len(conv['output']) for conv in final_dataset]):.0f} characters")
print(f"Average question length: {np.mean([len(conv['instruction']) for conv in final_dataset]):.0f} characters")
print(f"Longest response: {max([len(conv['output']) for conv in final_dataset])} characters")
print(f"Shortest response: {min([len(conv['output']) for conv in final_dataset])} characters")