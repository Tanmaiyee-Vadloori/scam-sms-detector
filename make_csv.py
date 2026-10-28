import csv

indian_train = [
    # Scams (label 1)
    ("Dear customer, your bank KYC has expired. Update within 24 hours or your account will be suspended: <link>", 1),
    ("Your electricity connection will be disconnected tonight at 9:30 PM due to unpaid bill. Call the officer immediately on <number>.", 1),
    ("Your courier is held at customs. Pay Rs 49 customs fee via UPI to release your parcel: <link>", 1),
    ("Work from home job! Earn Rs 3000-8000 daily by liking videos. No experience needed. WhatsApp HR on <number>.", 1),
    ("Congratulations! Your mobile number has won Rs 25 lakh in the lucky draw. Pay Rs 5,000 processing fee to claim.", 1),
    ("Your PAN card is not linked to Aadhaar. Your account will be frozen today. Link now: <link>", 1),
    ("Your credit card reward points worth Rs 7,850 expire today. Redeem now: <link>", 1),
    ("Pre-approved instant loan of Rs 5,00,000 at 0% interest. No documents required. Apply now: <link>", 1),
    ("Hi, I sent Rs 2,000 to your UPI by mistake. Please send it back, I am in urgent need.", 1),
    ("Join our VIP stock tips WhatsApp group. Guaranteed 300% returns in 1 month. Limited seats: <link>", 1),
    ("Your SIM card will be blocked in 2 hours due to incomplete e-KYC. Call customer care on <number> now.", 1),
    ("Income tax refund of Rs 15,490 has been approved. Verify your bank details to receive it: <link>", 1),
    # Normal messages (label 0)
    ("Rs 450.00 debited from A/c XX1234 on 28-10 to a UPI merchant. If this was not you, contact your bank.", 0),
    ("123456 is your OTP to verify your account. It is valid for 5 minutes. Never share your OTP.", 0),
    ("Your food order is out for delivery and will reach you in 15 minutes.", 0),
    ("Bro, send me the notes for tomorrow's DSA quiz pls", 0),
    ("Did you reach the hostel? Call me when you are free.", 0),
    ("Your PNR 4521873690 is confirmed. Train departs at 06:15 from platform 3.", 0),
    ("Class rescheduled: Machine Learning lecture moved to 11 AM in room F-102 tomorrow.", 0),
    ("Recharge of Rs 299 successful. Your plan is valid for 28 days.", 0),
    ("Your cab is arriving in 3 minutes. Driver: Ramesh, white sedan.", 0),
    ("Paid you back the 250 for dinner on UPI, check once", 0),
    ("Your appointment with Dr. Sharma is confirmed for Monday at 10:30 AM.", 0),
    ("Semester results are out. Check the student portal with your ID.", 0),
]

with open("indian_messages.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(indian_train)

print(f"Saved {len(indian_train)} messages to indian_messages.csv")

indian_test = [
    ("URGENT: Your bank account has been temporarily blocked. Complete KYC verification now to restore access: <link>", 1),
    ("Your gas subsidy is pending. Update your Aadhaar details today to receive Rs 1,200: <link>", 1),
    ("Dear candidate, you are shortlisted for a data entry job, Rs 25,000/month. Pay Rs 999 registration fee to confirm.", 1),
    ("Parcel delivery failed due to incomplete address. Update address and pay Rs 25 redelivery fee: <link>", 1),
    ("Your traffic challan of Rs 2,000 is unpaid. Pay now to avoid court action: <link>", 1),
    ("This is a customs officer. A package in your name contains illegal items. Call immediately to avoid arrest.", 1),
    ("Rs 1,200 credited to A/c XX5678 by UPI. Available balance updated.", 0),
    ("Lab submission deadline extended to Sunday midnight.", 0),
    ("Your grocery order has been delivered. Rate your experience in the app.", 0),
    ("Are you coming for the Quark rehearsal at 6? We need one more person for the robotics demo.", 0),
    ("654321 is your OTP for payment of Rs 499. Do not share this OTP with anyone.", 0),
    ("Library book due tomorrow. Please return or renew it at the counter.", 0),
]

# Held-out test set: NEVER used for training, only for evaluation
with open("indian_test.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(indian_test)

print(f"Saved {len(indian_test)} held-out test messages to indian_test.csv")