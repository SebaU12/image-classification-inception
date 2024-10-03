import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        i = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            

            optimizer.zero_grad()  
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)  
            i += 1
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        print(f'Epoch {epoch}/{num_epochs-1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

    print("Entrenamiento finalizado")
    return model

def validate_model(model, dataloader, criterion, device):
    model.eval()  
    running_loss = 0.0
    
    with torch.no_grad():  
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
    
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss

