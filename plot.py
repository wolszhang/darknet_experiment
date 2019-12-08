er = {}
ba = {}
ws = {}
er['train_acc'] = []
er['val_acc'] = []
er['train_loss'] = []
er['val_loss'] = []

ws['train_acc'] = []
ws['val_acc'] = []
ws['train_loss'] = []
ws['val_loss'] = []

ba['train_acc'] = []
ba['val_acc'] = []
ba['train_loss'] = []
ba['val_loss'] = []
prev = ''
with open('log','r+') as inp:
	for line in inp:
		if line.startswith('ER'):
			flag = 1
		elif line.startswith('BA'):
			flag = 2
		elif line.startswith('WS'):
			flag = 3
		elif 'epoch' in line:
			train_log = line.split('val')[0]
			val_log = line.split('val')[1]
			if flag == 1:
				er['train_acc'].append(float(train_log[train_log.index('train_acc:')+11:train_log.index('train_acc:')+17]))
				er['train_loss'].append(float(train_log[train_log.index('loss:')+6:train_log.index('loss:')+14]))
				er['val_acc'].append(float(val_log[val_log.index('_acc:')+5:val_log.index(' l')]))
				er['val_loss'].append(float(val_log[val_log.index('loss:')+6:val_log.index('loss:')+13]))
				
			elif flag == 2:
				ba['train_acc'].append(float(train_log[train_log.index('train_acc:')+11:train_log.index('train_acc:')+17]))
				ba['train_loss'].append(float(train_log[train_log.index('loss:')+6:train_log.index('loss:')+14]))
				ba['val_acc'].append(float(val_log[val_log.index('_acc:')+5:val_log.index(' l')]))
				ba['val_loss'].append(float(val_log[val_log.index('loss:')+6:val_log.index('loss:')+13]))
			else:
				ws['train_acc'].append(float(train_log[train_log.index('train_acc:')+11:train_log.index('train_acc:')+17]))
				ws['train_loss'].append(float(train_log[train_log.index('loss:')+6:train_log.index('loss:')+14]))
				ws['val_acc'].append(float(val_log[val_log.index('_acc:')+5:val_log.index(' l')]))
				ws['val_loss'].append(float(val_log[val_log.index('loss:')+6:val_log.index('loss:')+13]))
		
er['time'] = 48291.95084166527
ba['time'] = 47349.33334302902
ws['time'] = 33417

import matplotlib.pyplot as plt

plt.plot(range(100),er['train_acc'],label='Erdos-Renyi model')
plt.plot(range(100),ba['train_acc'],label='Barabasi-Albert model')
plt.plot(range(100),ws['train_acc'],label='Watts-Strogatz model')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.show()
plt.plot(range(100),er['train_loss'],label='Erdos-Renyi model')
plt.plot(range(100),ba['train_loss'],label='Barabasi-Albert model')
plt.plot(range(100),ws['train_loss'],label='Watts-Strogatz model')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()
plt.show()
plt.plot(range(100),er['val_acc'],label='Erdos-Renyi model')
plt.plot(range(100),ba['val_acc'],label='Barabasi-Albert model')
plt.plot(range(100),ws['val_acc'],label='Watts-Strogatz model')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('validation acc')
plt.show()
plt.show()
plt.plot(range(100),er['val_loss'],label='Erdos-Renyi model')
plt.plot(range(100),ba['val_loss'],label='Barabasi-Albert model')
plt.plot(range(100),ws['val_loss'],label='Watts-Strogatz model')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.show()

plt.bar([1,2,3], [er['time'], ba['time'], ws['time']], width = 0.4, tick_label=['Erdos-Renyi model','Barabasi-Albert model','Watts-Strogatz model'])
plt.xlabel('model')
plt.ylabel('time')

plt.show()