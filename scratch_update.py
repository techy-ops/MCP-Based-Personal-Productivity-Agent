with open('tests/test_agent_natural_language.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'result = await agent.ainvoke("Delete task 12.")',
    'result = await agent.ainvoke("Delete task 12.", confirmed=True)'
)
c = c.replace(
    'result = await agent.ainvoke("Cancel event 8.")',
    'result = await agent.ainvoke("Cancel event 8.", confirmed=True)'
)
c = c.replace(
    'result = await agent.ainvoke("Delete note 3.")',
    'result = await agent.ainvoke("Delete note 3.", confirmed=True)'
)

with open('tests/test_agent_natural_language.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(c)
print('Updated 3 delete tests in test_agent_natural_language.py')
