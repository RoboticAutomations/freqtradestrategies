from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models.user import User, UserRole
from src.security import get_password_hash

def main():
    db: Session = SessionLocal()

    username = "admin"
    password = "admin"

    if db.query(User).filter(User.username == username).first():
        print("Admin user already exists")
        return

    user = User(
        username=username,
        password_hash=get_password_hash(password),
        role=UserRole.ADMIN,
    )

    db.add(user)
    db.commit()
    print("Admin user created: admin / admin")

if __name__ == "__main__":
    main()
